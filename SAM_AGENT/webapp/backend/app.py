"""
SAM-Agent Web Application Backend
Provides real-time streaming conversation interface with WebSocket support
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime
import pandas as pd

# Add project root directory to system path
BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from AI_agents.SAMAgent import SAMMultiAIAgent

app = FastAPI(title="SAM Agent API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
frontend_dir = WEBAPP_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Pydantic models
class MessageModel(BaseModel):
    role: str
    content: str
    molecules: Optional[List[str]] = []
    timestamp: Optional[str] = None

class RestoreSessionRequest(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

class ClearSessionRequest(BaseModel):
    session_id: Optional[str] = None

# Global Agent instance
agent = None
conversation_history = []


def parse_agent_output(output_text: str) -> List[Dict[str, Any]]:
    """
    Parse verbose agent stdout and extract Thought / Action / Observation steps.
    This is intentionally lightweight: it targets the current structured-chat
    output format without trying to fully parse every possible LangChain log.
    """
    if not output_text or not isinstance(output_text, str):
        return []

    cleaned_text = re.sub(r'\x1b\[[0-9;]*m', '', output_text)
    cleaned_text = cleaned_text.replace('\r\n', '\n').replace('\r', '\n')

    steps: List[Dict[str, Any]] = []

    block_pattern = re.compile(
        r'Thought:\s*(.*?)\s*Action:\s*```\s*(\{.*?\})\s*```(?:\s*Observation:\s*(.*?))?(?=\n\s*Thought:|\n\s*Final Answer:|$)',
        re.DOTALL,
    )

    for match in block_pattern.finditer(cleaned_text):
        thought = match.group(1).strip()
        action_blob = match.group(2).strip()
        observation = (match.group(3) or '').strip()

        if thought:
            steps.append({
                "type": "thought",
                "content": thought,
                "timestamp": datetime.now().isoformat(),
            })

        try:
            action_data = json.loads(action_blob)
            steps.append({
                "type": "action",
                "tool": action_data.get("action", "Unknown"),
                "input": action_data.get("action_input", {}),
                "timestamp": datetime.now().isoformat(),
            })
        except json.JSONDecodeError:
            pass

        if observation:
            steps.append({
                "type": "observation",
                "content": observation,
                "timestamp": datetime.now().isoformat(),
            })

    if steps:
        return steps

    thought_pattern = re.compile(r'Thought:\s*(.*?)(?=\n\s*Action:|$)', re.DOTALL)
    action_pattern = re.compile(r'Action:\s*```\s*(\{.*?\})\s*```', re.DOTALL)

    thoughts = [item.strip() for item in thought_pattern.findall(cleaned_text) if item.strip()]
    actions = action_pattern.findall(cleaned_text)

    for i, thought in enumerate(thoughts):
        steps.append({
            "type": "thought",
            "content": thought,
            "timestamp": datetime.now().isoformat(),
        })

        if i < len(actions):
            try:
                action_data = json.loads(actions[i])
                steps.append({
                    "type": "action",
                    "tool": action_data.get("action", "Unknown"),
                    "input": action_data.get("action_input", {}),
                    "timestamp": datetime.now().isoformat(),
                })
            except json.JSONDecodeError:
                pass

    return steps


def is_valid_smiles(smiles: str) -> bool:
    """
    Basic SMILES validation without rdkit dependency
    """
    if not smiles or not isinstance(smiles, str):
        return False
    smiles = smiles.strip('.,;:\'')
    if len(smiles) < 10 or len(smiles) > 300:
        return False
    has_ring = bool(re.search(r'[cnops]\d', smiles, re.IGNORECASE))
    has_brackets = smiles.count('(') >= 1 or smiles.count('[') >= 1
    if not (has_ring or has_brackets):
        return False
    if smiles.count('(') != smiles.count(')') or smiles.count('[') != smiles.count(']'):
        return False
    if not any(elem in smiles for elem in ['C', 'N', 'O', 'P', 'S', 'c', 'n', 'o', '#']):
        return False
    if re.search(r'[\u4e00-\u9fff]', smiles):
        return False
    if smiles.startswith('http'):
        return False
    return True


def extract_molecules_from_steps(intermediate_steps) -> List[str]:
    """
    直接从 AgentExecutor 的 intermediate_steps 中提取 SMILES。
    优先级最高，完全不依赖正则解析文本。

    intermediate_steps 格式: List[(AgentAction, tool_output)]
    - Molecular_Generator 返回: (results_df, csv_return_value)
    - Property_Predictor  返回: dict{HOMO: df, LUMO: df, DM: df}
    """
    molecules = []
    seen = set()

    if not intermediate_steps:
        return molecules

    for action, observation in intermediate_steps:
        tool_name = getattr(action, 'tool', '')
        print(f"[extract_molecules_from_steps] tool={tool_name}, obs_type={type(observation).__name__}")
        if isinstance(observation, tuple):
            print(f"  tuple len={len(observation)}, elem0 type={type(observation[0]).__name__}")
        elif isinstance(observation, dict):
            print(f"  dict keys={list(observation.keys())}")
        else:
            print(f"  value preview={str(observation)[:100]}")

        # Molecular_Generator: returns (DataFrame, csv_result)
        if tool_name == 'Molecular_Generator':
            try:
                if isinstance(observation, tuple) and len(observation) >= 1:
                    df = observation[0]
                    if isinstance(df, pd.DataFrame) and 'smiles' in df.columns:
                        for smi in df['smiles'].dropna().tolist():
                            smi = str(smi).strip()
                            if smi and smi not in seen:
                                seen.add(smi)
                                molecules.append(smi)
                        print(f"[extract_molecules_from_steps] Generator: {len(molecules)} SMILES extracted")
            except Exception as e:
                print(f"[extract_molecules_from_steps] Generator parse error: {e}")

        # Property_Predictor: returns dict{HOMO/LUMO/DM: numpy array}
        # SMILES come from generated_data.csv or smiles_data.csv
        elif tool_name == 'Property_Predictor':
            try:
                # The predictor reads SMILES from CSV files, read them directly
                csv_paths = [
                    GENERATED_DATA_CSV,
                    RUNTIME_INPUT_CSV,
                ]
                for csv_path in csv_paths:
                    if csv_path.exists():
                        try:
                            df = pd.read_csv(csv_path)
                            col = 'SMILES' if 'SMILES' in df.columns else ('smiles' if 'smiles' in df.columns else None)
                            if col:
                                for smi in df[col].dropna().tolist():
                                    smi = str(smi).strip()
                                    if smi and smi not in seen:
                                        seen.add(smi)
                                        molecules.append(smi)
                                if molecules:
                                    print(f"[extract_molecules_from_steps] Predictor: {len(molecules)} SMILES from {csv_path}")
                                    break
                        except Exception:
                            pass
            except Exception as e:
                print(f"[extract_molecules_from_steps] Predictor parse error: {e}")

    return molecules


def extract_molecules(text: str) -> List[str]:
    """
    Fallback: extract SMILES from text using regex strategies.
    Used only when intermediate_steps yields no results.
    """
    if not text or not isinstance(text, str):
        return []
    
    molecules = []
    
    try:
        # Strategy 1: numbered list format  "1. SMILES"
        lines = text.split('\n')
        for i, line in enumerate(lines):
            num_match = re.match(r'^\s*(\d+)[.\)\]]\s*([A-Z][A-Za-z0-9\(\)\[\]=\#\-\+@\\/c]+)', line.strip())
            if num_match:
                potential_smiles = num_match.group(2).strip()
                if not re.search(r'[cnops]\d', potential_smiles):
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if not re.match(r'^[\d\-\u2013]', next_line):
                            potential_smiles += next_line.split('-')[0].strip()
                potential_smiles = re.split(
                    r'\s*[-\u2013]\s*(?:HOMO|LUMO|Dipole|Band|Efficiency|Molecule)',
                    potential_smiles
                )[0].strip()
                if is_valid_smiles(potential_smiles):
                    molecules.append(potential_smiles)
        
        # Strategy 2: SMILES before property keywords
        if len(molecules) < 2:
            pattern = r'([A-Z][A-Za-z0-9\(\)\[\]=\#\-\+@\\/c]{20}?)(?:\s*[-\u2013]\s*(?:HOMO|LUMO|Dipole|Molecule|Properties)|\n\s*[-\u2013]|\n\s*\d+[\.)\]])'
            for match in re.findall(pattern, text):
                if is_valid_smiles(match):
                    molecules.append(match)

        # Strategy 3: SMILES in backticks
        if len(molecules) < 1:
            for match in re.findall(r'`([A-Za-z0-9\(\)\[\]=\#\-\+@\\/]{10})`', text):
                if is_valid_smiles(match):
                    molecules.append(match)
        
        # Strategy 4: SMILES after explicit label
        if len(molecules) < 1:
            for match in re.findall(r'(?:SMILES|smiles)[:\s=]+([A-Za-z0-9\(\)\[\]=\#\-\+@\\/]{10})', text):
                clean = match.strip('.,;:\' "')
                if is_valid_smiles(clean):
                    molecules.append(clean)

        # Deduplicate
        seen = set()
        unique = []
        for mol in molecules:
            if mol not in seen:
                seen.add(mol)
                unique.append(mol)
        return unique[:10]
    
    except Exception as e:
        print(f"Error extracting molecules: {e}")
        return []


def summarize_tool_observation(observation: Any) -> Dict[str, Any]:
    if isinstance(observation, Exception):
        return {"status": "error", "preview": str(observation)}
    if isinstance(observation, tuple):
        parts = [summarize_tool_observation(item)["preview"] for item in observation[:2]]
        return {"status": "success", "preview": " | ".join(parts)}
    if isinstance(observation, dict):
        return {"status": "success", "preview": str({key: type(value).__name__ for key, value in observation.items()})}
    if isinstance(observation, pd.DataFrame):
        return {"status": "success", "preview": f"DataFrame rows={len(observation)}, columns={list(observation.columns)}"}
    preview = str(observation)
    lowered_preview = preview.lower()
    status = "error" if (
        lowered_preview.startswith("error")
        or "invalid scaffold" in lowered_preview
        or "not found" in lowered_preview
    ) else "success"
    return {"status": status, "preview": preview[:500]}


def build_tool_trace(intermediate_steps) -> List[Dict[str, Any]]:
    trace = []
    for index, (action, observation) in enumerate(intermediate_steps, start=1):
        summary = summarize_tool_observation(observation)
        trace.append({
            "index": index,
            "tool": getattr(action, "tool", "Unknown"),
            "input": getattr(action, "tool_input", {}),
            "status": summary["status"],
            "output_preview": summary["preview"],
            "timestamp": datetime.now().isoformat(),
        })
    return trace


@app.on_event("startup")
async def startup_event():
    """
    Initialize Agent on startup
    """
    global agent
    
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    deepseek_key = os.getenv('DeepSeek_API_KEY')
    
    try:
        agent = SAMMultiAIAgent(
            open_ai_key=openai_key,
            deepseek_key=deepseek_key,
            tavily_key=tavily_key,
            llm_model='chatgpt',
            verbose=True
        )
        print("\u2713 SAM Agent initialized successfully")
    except Exception as e:
        print(f"\u2717 Agent initialization failed: {e}")
        agent = None


@app.get("/")
async def root():
    frontend_path = WEBAPP_DIR / "frontend" / "index.html"
    return FileResponse(str(frontend_path))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time conversation
    """
    await websocket.accept()
    
    if agent is None:
        await websocket.send_json({
            "type": "error",
            "message": "Agent not initialized, please check API keys configuration"
        })
        await websocket.close()
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            session_id = data.get("session_id", "default")
            
            if not user_message:
                continue
            
            print(f"\n\U0001f4e8 Received message [Session: {session_id}]: {user_message[:50]}{'...' if len(user_message) > 50 else ''}")
            
            conversation_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            await websocket.send_json({
                "type": "user_message",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            await websocket.send_json({
                "type": "thinking_start",
                "timestamp": datetime.now().isoformat()
            })
            
            try:
                import io
                from contextlib import redirect_stdout
                
                output_buffer = io.StringIO()
                loop = asyncio.get_event_loop()
                
                def run_agent():
                    with redirect_stdout(output_buffer):
                        return agent.invoke(user_message, session_id=session_id)
                
                response = await loop.run_in_executor(None, run_agent)
                
                # Capture verbose stdout only as a fallback/debug source. The UI now uses structured
                # tool traces from intermediate_steps instead of incomplete verbose reasoning text.
                verbose_output = output_buffer.getvalue() or ""
                steps = parse_agent_output(verbose_output)
                
                # Parse response
                response_str = ""
                intermediate_steps = []
                if isinstance(response, dict):
                    response_str = str(response.get("output", "") or "")
                    intermediate_steps = response.get("intermediate_steps", [])
                elif isinstance(response, str):
                    response_str = response
                else:
                    response_str = str(response) if response else ""

                real_thoughts = [
                    step.get("content", "")
                    for step in steps
                    if step.get("type") == "thought" and step.get("content")
                ]

                tool_trace = build_tool_trace(intermediate_steps)
                if tool_trace or real_thoughts:
                    await websocket.send_json({
                        "type": "tool_trace",
                        "steps": tool_trace,
                        "thoughts": real_thoughts,
                        "timestamp": datetime.now().isoformat()
                    })

                # --- Primary: extract SMILES directly from tool outputs ---
                print(f"[Debug] intermediate_steps count: {len(intermediate_steps)}")
                for i, (act, obs) in enumerate(intermediate_steps):
                    print(f"  step{i}: tool={getattr(act,'tool','?')}, obs_type={type(obs).__name__}, obs_preview={str(obs)[:80]}")
                molecules = extract_molecules_from_steps(intermediate_steps)
                print(f"[Molecules] From intermediate_steps: {len(molecules)} found")

                # --- Fallback: regex on final response text ---
                if not molecules and response_str.strip():
                    try:
                        is_synthesis = any(kw in response_str.lower() for kw in [
                            'retrosynthesis', 'retrosynthetic', 'precursor',
                            'synthesis route', 'synthetic route'
                        ])
                        if not is_synthesis:
                            molecules = extract_molecules(response_str)
                            print(f"[Molecules] Fallback regex: {len(molecules)} found")
                    except Exception as e:
                        print(f"Error extracting molecules: {e}")
                        molecules = []
                
                await websocket.send_json({
                    "type": "assistant_message",
                    "content": response_str,
                    "molecules": molecules,
                    "steps": tool_trace,
                    "thoughts": real_thoughts,
                    "timestamp": datetime.now().isoformat()
                })
                
                conversation_history.append({
                    "role": "assistant",
                    "content": response_str,
                    "molecules": molecules,
                    "steps": tool_trace,
                    "thoughts": real_thoughts,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/api/history")
async def get_history():
    return {
        "history": conversation_history,
        "count": len(conversation_history)
    }


@app.post("/api/clear")
async def clear_history(request: ClearSessionRequest = Body(default=None)):
    """
    Clear conversation history (including Agent's chat memory)
    """
    global conversation_history, agent
    
    if request and request.session_id:
        session_id = request.session_id
        if agent:
            agent.clear_session_memory(session_id)
            print(f"\u2713 Cleared memory for session {session_id}")
        return {"message": f"Memory for session {session_id} has been cleared"}
    else:
        conversation_history = []
        if agent:
            agent.session_memories.clear()
            print("\u2713 All session memories cleared")
        return {"message": "All conversation history has been cleared"}


@app.post("/api/restore_session")
async def restore_session(request: RestoreSessionRequest):
    """
    Restore session memory from frontend history
    """
    global agent
    
    if not agent:
        return {"success": False, "message": "Agent not initialized"}
    if not request.session_id:
        return {"success": False, "message": "Missing session_id"}
    
    try:
        agent.restore_session_memory(request.session_id, request.messages)
        return {
            "success": True,
            "message": f"Successfully restored memory for session {request.session_id}",
            "message_count": len(request.messages)
        }
    except Exception as e:
        print(f"Failed to restore session memory: {e}")
        return {"success": False, "message": str(e)}


@app.get("/api/tools")
async def get_tools():
    tools = [
        {"name": "Molecular_Generator", "description": "Molecular Generator - Generate SAM molecules based on scaffolds and anchoring groups", "icon": "🧬"},
        {"name": "Property_Predictor", "description": "Property Predictor - Predict HOMO, LUMO and dipole moment of molecules", "icon": "📊"},
        {"name": "Device_Efficiency_Evaluator", "description": "Device Efficiency Evaluator - Evaluate perovskite solar cell efficiency", "icon": "⚡"},
        {"name": "Molecular_Informatics_Tools", "description": "Molecular Informatics Tools - SMILES conversion, visualization, etc.", "icon": "🔬"},
        {"name": "Retrosynthesis_Planner", "description": "Retrosynthesis Planner - Generate molecular synthesis routes", "icon": "🔄"},
        {"name": "RetrievalQA", "description": "RAG Retrieval - Retrieve information from literature database", "icon": "📚"},
        {"name": "Supplier_info", "description": "Supplier Info Query - Query chemical suppliers", "icon": "🏪"},
        {"name": "Literature_Extractor", "description": "Literature Extractor - Identify SAM systems from literature PDFs and generate structured knowledge cards", "icon": "📄"},
        {"name": "python_executor", "description": "Python Code Executor - Execute computational tasks", "icon": "🐍"},
        {"name": "TavilySearchResults", "description": "Web Search - Search for latest information", "icon": "🔍"}
    ]
    return {"tools": tools}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

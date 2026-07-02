import logging
# Set the logging level for httpx to WARNING to suppress INFO messages
logging.getLogger("httpx").setLevel(logging.WARNING)

from langchain_core.prompts import ChatPromptTemplate
from AI_agents.config.llm import get_chat_model_config
from AI_agents.tools.molecular_generator.sam_generator import generator_tool
from AI_agents.tools.property_predictor.prop_predictor import Predictor
from AI_agents.tools.molecular_informatics_tools.Price import Molinfo
from AI_agents.tools.RAG.retrieval import RetrievalQA
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.tools.base import StructuredTool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from AI_agents.tools.molecular_informatics_tools.mol_agent import Mol_agent
from AI_agents.tools.literature_extractor import LiteratureExtractor
from AI_agents.tools.retrosynthesis_planner.advisor import plan_and_print
from AI_agents.tools.device_evaluator.device_evaluator import DeviceEvaluator
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List,Optional,Union

###################################################
class GeneratorToolInput(BaseModel):
    gen_size: int
    scaf_condition: List[str]
    anchoring_group: Optional[str] = 'O=P(O)(O)'
    
class PredictorToolInput(BaseModel):
    smiles: Optional[List[str]] = None
    generated: bool
    HOMO: bool
    LUMO: bool
    DM: bool

class MolinfoToolInput(BaseModel):
    generated: bool
    mol_list: Optional[List[str]] = None
    
class Mol_agentInput(BaseModel):
    input_message: str
    
class LiteratureExtractorInput(BaseModel):
    path: str
    
class RetrievalQAInput(BaseModel):
    query: str
    top_k: int
    
class SynthesisInput(BaseModel):
    smiles:str
    
class DeviceEvaluatorInput(BaseModel):
    smiles: Union[str, List[str]]  # 支持单个或多个 SMILES
    Cs_ratio: float
    FA_ratio: float
    Br_ratio: float
    PVK_Eg: float
    ITO: int
    FTO: int
    NiO: int
    SnO2: int
    ZnO: int
    TiO2: int
###################################################
    


class SAMMutiAIAgent:
    def __init__(self,open_ai_key,deepseek_key ,tavily_key=None,verbose=True,llm_model='deepseek'):
        self.open_ai_key=open_ai_key
        self.tavily_key=tavily_key
        self.deepseek_key = deepseek_key
        self.verbose = verbose
        self.input=None
        self.llm_model = llm_model
        self.prop_predictor = Predictor()
        # Instantiate tools here
        self.molinfo_tool = Molinfo()
        self.mol_agent_tool = Mol_agent(self.open_ai_key, verbose=True)
        self.literature_extractor_tool = LiteratureExtractor(open_ai_key=self.open_ai_key, deepseek_key=self.deepseek_key, llm_model=self.llm_model, verbose=True)
        self.retrieval_qa_tool = RetrievalQA(llm='chatgpt', open_ai_key=self.open_ai_key, deepseek_key=self.deepseek_key)
        self.device_evaluator = None
        self.python_repl = PythonREPL()
        self.chat_history = None
        # Maintain independent memory for each session
        self.session_memories = {}  # {session_id: InMemoryChatMessageHistory}
        self._tools = None
        self._agent_executor = None


    
    # Call the LLM model 
    def llm(self):
        engine = self.llm_model.lower()
        config = get_chat_model_config(engine, purpose="agent")
        if engine == 'chatgpt':
            api_key = self.open_ai_key
            key_name = "OPENAI_API_KEY"
        elif engine == 'deepseek':
            api_key = self.deepseek_key
            key_name = "DeepSeek_API_KEY"
        else:
            raise ValueError("Unsupported LLM model. Please choose 'chatgpt' or 'deepseek'.")
        if not api_key:
            raise ValueError(f"{key_name} is required for {engine} LLM calls.")
        return ChatOpenAI(**config, api_key=api_key)

    @staticmethod
    def _web_search_unavailable(*args, **kwargs):
        return "Web search is unavailable because TAVILY_API_KEY is not configured. Please set TAVILY_API_KEY to enable this tool."
    
    # def debug_generator_tool(*args, **kwargs):
    #     print("generator_tool called with args:", args)
    #     print("generator_tool called with kwargs:", kwargs)
    #     # Validate input
    #     if not kwargs:
    #         raise ValueError("Tool input must be a dictionary with the required keys.")
    #     if "gen_size" not in kwargs or "scaf_condition" not in kwargs or "anchoring_group" not in kwargs:
    #         raise ValueError("Missing one or more required keys: 'gen_size', 'scaf_condition', 'anchoring_group'.")
    #     return generator_tool(*args, **kwargs)
    
    
    def _device_evaluator_wrapper(self, smiles: Union[str, List[str]], 
                                   Cs_ratio: float, FA_ratio: float, 
                                   Br_ratio: float, PVK_Eg: float, 
                                   ITO: int, FTO: int, NiO: int, 
                                   SnO2: int, ZnO: int, TiO2: int) -> str:
        """
        Wrapper function: device efficiency evaluation supporting single or multiple SMILES
        """
        # Lazily load the evaluator so model-loading issues do not prevent the whole Agent from starting.
        try:
            if self.device_evaluator is None:
                self.device_evaluator = DeviceEvaluator()
        except Exception as e:
            return f"器件评估工具暂时不可用：{str(e)}"

        # Convert to list for unified processing
        smiles_list = [smiles] if isinstance(smiles, str) else smiles
        
        results = []
        for smi in smiles_list:
            try:
                result = self.device_evaluator.predict(
                    smiles=smi,
                    Cs_ratio=Cs_ratio,
                    FA_ratio=FA_ratio,
                    Br_ratio=Br_ratio,
                    PVK_Eg=PVK_Eg,
                    ITO=ITO,
                    FTO=FTO,
                    NiO=NiO,
                    SnO2=SnO2,
                    ZnO=ZnO,
                    TiO2=TiO2
                )
                results.append(result if len(smiles_list) == 1 else f"{smi}: {result}")
            except Exception as e:
                results.append(f"Error: {str(e)}" if len(smiles_list) == 1 else f"{smi}: Error: {str(e)}")
        
        return "\n".join(results)
    
    def bind_tools(self):
        if self._tools is not None:
            return self._tools

        generator = StructuredTool(
        name="Molecular_Generator",
        description=(
        "This tool is strictly for generating new molecules. "
        "The scaf_condition values must be valid scaffold SMILES, not molecule names. "
        "If the current scaffold is only available as a molecule name, obtain or infer its SMILES "
        "before calling this tool. "
        "Both scaffold and anchoring group are SMILES. "
        "The anchoring_group is optional - if not specified, it will default to phosphonic "
        "acid group 'O=P(O)(O)' (as used in SAM-GPT training data). "
        "Do not use this tool for general questions or explanations."),
        args_schema=GeneratorToolInput,
        func=generator_tool,
        )
        
        prop_predictor= StructuredTool(
            name='Property_Predictor',
            description=(
                "This tool is strictly for predicting the property (HOMO, LUMO, and Dipole moment) of molecules."
                "The data is already stored in the dataframe"
                "Do not use this tool for general questions or explanations."
                ),
            args_schema=PredictorToolInput,
            func=self.prop_predictor.prop_pred
        )

            
        repl_tool = Tool(
            name="python_executor",
            description=("A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
                         "When you find a task that you can not solve based on provided functions, you can write some codes to solve tasks like complex calculation, creating a machine learning models."),
            func=self.python_repl.run,
        )
        
        if self.tavily_key:
            search = TavilySearchResults(max_results=5, 
                                         tavily_api_key=self.tavily_key,
                                         search_depth = "advanced",
                                         include_answer= True,
                                         description="A search engine optimized for comprehensive, accurate, and trusted results."
                                         "When the task is beyond your capability. You should use this tool to help you find final solution."
                                         "Input should be a search query.")
        else:
            search = Tool(
                name="tavily_search_results_json",
                description="Web search placeholder. Use only to explain that web search requires TAVILY_API_KEY.",
                func=self._web_search_unavailable,
            )
        
        
        Mol_info=StructuredTool(
            name="Supplier_info",
            description=("This tool is for search for the molecular commercial informations, including its price, supplier,purity ...."
                         "Use this to if you need some commercial information about the molecules."
                         "The input should be SMILES or molecule name, if you received none of them, the smiles input should be None and you should use generated dataset."
                         ),
            args_schema=MolinfoToolInput,
            func=self.molinfo_tool.collect_info

                   
        )
        
    
        Molecular_informatics_tools=StructuredTool(
            name="Molecular_Informatics_Tools",
            description=("This tool is strictly for answering the questions about the molecules. "
                         "Use it to convert molecule names to SMILES, SMILES to IUPAC names, "
                         "visualize molecules, and draw chemical reactions. "
                         "Just use format Products>SAScore>Intermediate products|Intermediate products>SAScore>Precursors or direcltly the Synthesis_advisor ouput(eg. C1=CC=C2C(=C1)C3=CC=CC=C3N2CCP(=O)(O)O>0.0010>c1ccc2c(c1)[nH]c1ccccc12.O=P(O)(O)CCCl), if you want to draw the reaction routes."
                         "Do not use this tool for general questions or explanations."),
            args_schema=Mol_agentInput,
            func=self.mol_agent_tool.invoke
            )

        Literature_extractor=StructuredTool(
            name="Literature_Extractor",
            description=("This tool extracts structured knowledge cards from scientific literature PDFs about self-assembled monolayers in perovskite solar cells."
                         "It identifies SAM systems in a paper, distinguishes single and mixed systems, and returns system-level knowledge cards."
                         "Only PDF file path input is supported."
                         "Do not use this tool for general questions or explanations."
                ),
            args_schema=LiteratureExtractorInput,
            func=self.literature_extractor_tool.extract_as_json
            )
        
        # RetrievalQA by RAG
        RetrievalQA_agent = StructuredTool(
            name="RetrievalQA",
            description=("This tool is used for RAG retrieval of relevant data for Self-assembled molecules (SAM) in PSC."
                         "Such as SAM synthesis, SAM properties, device fabrication, device performance"
                         "The top_k is the number of the relevant documents you want to retrieve."
                         "You should give all references to the user in the final response."
                         "The input is a query, and you need to modfiy the query standard to better matching the vector database."
            ),
            args_schema=RetrievalQAInput,
            func=self.retrieval_qa_tool.run,
        )
        
        # Retrosynthesis Planner
        retrosynthesis_planner = StructuredTool(
            name="Retrosynthesis_Planner",
            description=("This tool is used for synthesis planning of the molecules."
                         "The input should be SMILES of the target molecule."
                         "If you got nothing, you can try RetrievalQA_agent."
            ),
            args_schema=SynthesisInput,
            func=plan_and_print,
        )
        
        # Device Efficiency Evaluator
        device_evaluator_tool = StructuredTool(
            name="Device_Efficiency_Evaluator",
            description=(
                "This tool predicts the efficiency class of perovskite solar cell devices: low, medium, or high efficiency.\n"
                "\n"
                "【REQUIRED PARAMETERS】:\n"
                "1. smiles: SMILES string (or list of SMILES) of the SAM molecule(s) (required)\n"
                "2. Perovskite composition parameters (all 4 required):\n"
                "   - Cs_ratio: Cesium ion ratio, range 0-1, e.g., 0.05 for 5%\n"
                "   - FA_ratio: Formamidinium ion ratio, range 0-1, e.g., 0.9 for 90%\n"
                "   - Br_ratio: Bromine ion ratio, range 0-1, e.g., 0.15 for 15%\n"
                "   - PVK_Eg: Perovskite bandgap in eV, typical range 1.5-1.7, e.g., 1.56\n"
                "3. Substrate type (choose 1 from 6, set chosen to 1, others to 0):\n"
                "   - ITO: Indium tin oxide substrate\n"
                "   - FTO: Fluorine-doped tin oxide substrate\n"
                "   - NiO: Nickel oxide substrate\n"
                "   - SnO2: Tin oxide substrate\n"
                "   - ZnO: Zinc oxide substrate\n"
                "   - TiO2: Titanium oxide substrate\n"
                "\n"
                "【PRE-CALL CHECK】:\n"
                "⚠️ DO NOT call this tool if the user has not provided all required parameters.\n"
                "Ask the user for missing parameters first with examples.\n"
                "\n"
                "【EXAMPLE QUESTION TO ASK USER】:\n"
                "To predict device efficiency, I need the following information:\n"
                "1. Perovskite composition: Cs/FA/Br ion ratios and bandgap\n"
                "   Example: 5% Cs (Cs_ratio=0.05), 90% FA (FA_ratio=0.9), 15% Br (Br_ratio=0.15), bandgap 1.56eV\n"
                "2. Substrate type (choose one): ITO/FTO/NiO/SnO2/ZnO/TiO2\n"
                "   Example: NiO substrate (NiO=1, others=0)\n"
                "\n"
                "【PARAMETER EXTRACTION GUIDE】:\n"
                "Extract parameter values from user's natural language responses:\n"
                "- '5% Cs' or 'Cs 5%' or 'cesium 5%' → Cs_ratio=0.05\n"
                "- '90% FA' or 'formamidinium 90%' → FA_ratio=0.9\n"
                "- '15% Br' or 'bromine 15%' → Br_ratio=0.15\n"
                "- 'bandgap 1.56' or '1.56 eV' → PVK_Eg=1.56\n"
                "- 'NiO substrate' or 'nickel oxide' or 'use NiO' → NiO=1, ITO=0, FTO=0, SnO2=0, ZnO=0, TiO2=0\n"
                "\n"
                "【OUTPUT】: Only the predicted efficiency class: low, medium, or high efficiency. Do not report confidence scores, estimated PCE ranges, averages, or vote distributions."
            ),
            args_schema=DeviceEvaluatorInput,
            func=self._device_evaluator_wrapper
        )
        
        
        self._tools = [generator,prop_predictor,repl_tool,Mol_info,
                search,Molecular_informatics_tools,Literature_extractor,
                RetrievalQA_agent,retrosynthesis_planner,device_evaluator_tool]
        return self._tools
    
    
    
    def prompt(self):
        system_prompt = """You are a materials science assitant in photovoltatic area, If you are not sure about answer to the user's request, use your tools to gather the relevant 
        information and you need to give detail information for subagent to execute: do NOT guess or make up an answer.
        If you are uncertain about the request you can ask user again. Also, for your final response, you should give user some suggestion based on your capability.
        You have access to the following tools:
        
        {tools}

        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

        Valid "action" values: "Final Answer" or {tool_names}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }}
        ```

        Follow this format:

        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        ```
        {{
        "action": "Final Answer",
        "action_input": "Final response to human"
        }}

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Thought, Action:```$JSON_BLOB```then Observation"""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("human", "{agent_scratchpad}"),
            ]
        )
        
        
        return prompt
    

    def agent(self):
        if self._agent_executor is not None:
            return self._agent_executor

        tools = self.bind_tools()
        structured_agent = create_structured_chat_agent(llm=self.llm(),
                                                        tools = tools,
                                                        prompt = self.prompt()
                                                        )
        self._agent_executor = AgentExecutor(
            agent = structured_agent,
            tools= tools,
            verbose = True, # some intermediate steps will be printed if True
            max_iterations=30,  # Increased from 15 to 30
            max_execution_time=600,  # Increased to 600 seconds (10 minutes) for slow property prediction
            handle_parsing_errors=True,
            return_intermediate_steps=True,  # 返回工具调用的原始结果，用于直接提取 SMILES
            )
    
        return self._agent_executor


    def get_session_memory(self, session_id: str):
        """
        Get or create memory for a specific session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            InMemoryChatMessageHistory: Memory object for this session
        """
        if session_id not in self.session_memories:
            print(f"🆕 Creating new session memory: {session_id}")
            self.session_memories[session_id] = InMemoryChatMessageHistory(session_id=session_id)
        return self.session_memories[session_id]
    
    def clear_session_memory(self, session_id: str):
        """
        Clear memory for a specific session
        
        Args:
            session_id (str): Session ID
        """
        if session_id in self.session_memories:
            self.session_memories[session_id].clear()
            print(f"🗑️ Cleared session memory: {session_id}")
    
    def restore_session_memory(self, session_id: str, messages: list):
        """
        Restore session memory from history message list
        
        Args:
            session_id (str): Session ID
            messages (list): History message list, each message format: {"role": "user/assistant", "content": "..."}
        """
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Clear existing memory (if any)
        if session_id in self.session_memories:
            self.session_memories[session_id].clear()
        
        # Get or create session memory
        memory = self.get_session_memory(session_id)
        
        # Add history messages to memory
        for msg in messages:
            if msg.get("role") == "user":
                memory.add_message(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                memory.add_message(AIMessage(content=msg.get("content", "")))
        
        print(f"♻️ Restored session {session_id} memory with {len(messages)} messages")
    
    def invoke(self, input_message: str, session_id: str = "default"):
        """
        Invoke the agent with an input message.
        Returns a dict with 'output' and 'intermediate_steps' so the backend
        can extract SMILES directly from tool outputs.

        Args:
            input_message (str): The input message for the agent.
            session_id (str): The session ID for maintaining conversation history.

        Returns:
            dict: {"output": str, "intermediate_steps": list}
        """
        from langchain_core.messages import HumanMessage, AIMessage

        # Get session memory
        memory = self.get_session_memory(session_id)

        print(f"\n{'='*60}")
        print(f"💬 Session ID: {session_id}")
        print(f"📝 Current conversation history length: {len(memory.messages)}")
        print(f"{'='*60}\n")

        # Build chat_history from memory messages
        chat_history = list(memory.messages)

        try:
            # Call AgentExecutor directly (not wrapped) to preserve intermediate_steps
            response = self.agent().invoke({
                "input": input_message,
                "chat_history": chat_history,
            })
        except Exception as e:
            print(f"Error invoking agent: {e}")
            return {"output": f"Error: {str(e)}", "intermediate_steps": []}

        # Manually update session memory
        output_text = ""
        if isinstance(response, dict):
            output_text = response.get("output", "") or ""
        else:
            output_text = str(response) if response else ""

        memory.add_message(HumanMessage(content=input_message))
        memory.add_message(AIMessage(content=output_text))

        print(f"\n✅ Conversation completed, current history length: {len(memory.messages)}")
        print(f"📊 Total active sessions: {len(self.session_memories)}")

        if isinstance(response, dict):
            return response  # contains 'output' and 'intermediate_steps'
        return {"output": output_text, "intermediate_steps": []}


SAMMultiAIAgent = SAMMutiAIAgent
        


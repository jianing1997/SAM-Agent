from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_structured_chat_agent
from AI_agents.tools.molecular_informatics_tools.mol_tool import Mol_tool
from langchain.tools.base import StructuredTool
from pydantic import BaseModel
from typing import List,Optional


class tools_Input(BaseModel):
    data_list: List[str]
    generated: bool = False
    
class image_tool_input(BaseModel):
    smiles_list: List[str]  
    generated: bool = False
    molsPerRow: Optional[int] = 5
    display_limit: Optional[int] = None

class reaction_image_input(BaseModel):
    reaction_routes_str: str
    


class Mol_agent:
    def __init__(self,open_ai_key,verbose=True):
        self.open_ai_key=open_ai_key
        self.verbose = verbose
        
    # Call the mol agent 
    def llm(self):
        model=ChatOpenAI(model_name="gpt-4.1-mini",
                         temperature=0.3,
                         max_tokens=None,
                         timeout=None,
                         base_url="https://api.chatanywhere.tech/v1",
                         max_retries=3,
                         api_key=self.open_ai_key
                         )
        return model
    
    def mol_tools(self):
        Name_SMILES=StructuredTool(
            name='IUPAC name to SMILES',
            description=(
                "This tool is used for converting IUPAC name or chemical names to SMILES"
            ),
            args_schema=tools_Input,
            func=Mol_tool().Name2Smiles
        )
        
        SMILES_Name=StructuredTool(
            name='SMILES to IUPAC name',
            description=(
                "This tool is used for converting SMILES to IUPAC name"
            ),
            args_schema=tools_Input,
            func=Mol_tool().Smiles2Name
        )
            
        Mol_Image=StructuredTool(
            name='Mol_visualization',
            description=(
                "This tool is used for display the molecular structures by image."
                "The data parameter is SMILES."
            ),
            args_schema=image_tool_input,
            func=Mol_tool().Mol2Image
        )
        
        Reaction_Image = StructuredTool(
            name='reaction_visualization',
            description=(
                "This tool is used for display the reaction routes by image."
                "The data parameter is string of reaction routes."
                "The format of the data should follow this: Products>SAScore>Intermediate products|Intermediate products>SAScore>Precursors."
                "Example:'O=C(OCCCCCCCCCCCCCP(=O)(O)O)c1ccccc1>0.8935>CCOP(=O)(CCCCCCCCCCCCCOC(=O)c1ccccc1)OCC|CCOP(=O)(CCCCCCCCCCCCCOC(=O)c1ccccc1)OCC>0.8282>O=C(OCCCCCCCCCCCCCBr)c1ccccc1.CCOP(OCC)OCC'."
            ),
            args_schema=reaction_image_input,
            func=Mol_tool().reaction2image
        )
        return [Name_SMILES,SMILES_Name,Mol_Image,Reaction_Image]
        
        
        
    
    def prompt(self):
        system = '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:

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

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Thought, Action:```$JSON_BLOB```then Observation'''

        human = '''{input}

        {agent_scratchpad}

        (reminder to respond in a JSON blob no matter what)'''

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", human),
            ]
        )
        return prompt
    
    def agent(self):
        mol_structured_agent = create_structured_chat_agent(llm=self.llm(),
                                                        tools = self.mol_tools(),
                                                        prompt = self.prompt()
                                                        )
        agent_executor = AgentExecutor(
            agent = mol_structured_agent,
            tools= self.mol_tools(),
            verbose = True, # some intermediate steps will be printed if True
            max_iterations=5,
            max_execution_time= 40,
            handle_parsing_errors=True,
            )
    
        return agent_executor

    def invoke(self, input_message: str):
        """
        Invoke the agent with an input message.
        Directly dispatches to the appropriate tool based on keywords,
        bypassing the sub-LLM to avoid dependency on unstable third-party proxies.

        Args:
            input_message (str): The input message for the agent.

        Returns:
            str: The output from the agent.
        """
        from AI_agents.tools.molecular_informatics_tools.Mol_utils import IUPAC_name2Smiles, Smiles2IUPAC_name
        from AI_agents.tools.molecular_informatics_tools.mol_tool import Mol_tool
        import re

        msg_lower = input_message.lower()
        mol_tool = Mol_tool()

        # --- Name to SMILES ---
        if any(kw in msg_lower for kw in ['to smiles', 'convert', 'smiles of', 'smiles for', 'get smiles', 'name to smiles', 'chemical name']):
            # Extract quoted names or the whole message as fallback
            names = re.findall(r'"([^"]+)"|\'([^\']+)\'', input_message)
            if names:
                name_list = [n[0] or n[1] for n in names]
            else:
                # Try to strip common command words and use the rest as the name
                cleaned = re.sub(
                    r'(?i)(convert|to smiles|get smiles|smiles of|smiles for|the smiles|please|chemical name of|chemical name for|what is|what\'s)',
                    '', input_message
                ).strip(' ?,.')
                name_list = [cleaned] if cleaned else [input_message]

            results = []
            for name in name_list:
                smiles = IUPAC_name2Smiles(name.strip())
                results.append(f"{name}: {smiles}")
            return '\n'.join(results)

        # --- SMILES to Name ---
        elif any(kw in msg_lower for kw in ['to iupac', 'iupac name', 'name of', 'smiles to name']):
            smiles_list = re.findall(r'[A-Za-z0-9@+\-\[\]\(\)=#\\/]{6,}', input_message)
            if smiles_list:
                results = mol_tool.Smiles2Name(smiles_list)
                return str(results)
            return "No SMILES found in input."

        # --- Visualization ---
        elif any(kw in msg_lower for kw in ['visuali', 'draw', 'show molecule', 'display molecule', 'image']):
            smiles_list = re.findall(r'[A-Za-z0-9@+\-\[\]\(\)=#\\/]{6,}', input_message)
            if smiles_list:
                return mol_tool.Mol2Image(smiles_list)
            return "No SMILES found to visualize."

        # --- Reaction visualization ---
        elif any(kw in msg_lower for kw in ['reaction', 'route', '>>', '>>']):
            return mol_tool.reaction2image(input_message)

        # --- Fallback: use sub-LLM agent ---
        else:
            try:
                agent_executor = self.agent()
                response = agent_executor.invoke({"input": input_message})
                return response.get("output")
            except Exception as e:
                return f"Error invoking Mol_agent: {str(e)}"
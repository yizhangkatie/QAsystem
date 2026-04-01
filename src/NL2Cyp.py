from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class NL2Cyp:
    def __init__(self, neo4j_url, neo4j_username, neo4j_password, nvidia_api_key):
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.nvidia_api_key = nvidia_api_key

        self.cypher_generation_prompt = self._create_cypher_prompt()
        self.qa_generation_prompt = self._create_qa_prompt()

        self.graph = self._initialize_graph()
        self.hospital_cypher_chain = self._initialize_chain()

    def _create_cypher_prompt(self):
        template = """
        任务：为Neo4j图数据库生成Cypher查询。
        说明：仅使用提供的模式中的关系类型和属性，不要使用任何未提供的其他关系类型或属性。
        模式：{schema}
        注意：不要回答任何可能要求你构建除Cypher语句之外的任何文本的问题。
        确保查询中的关系方向是正确的，确保正确地为实体和关系设置别名。
        不要运行会向数据库添加或删除内容的任何查询。
        问题是：{question}
        """
        return PromptTemplate(input_variables=["schema", "question"], template=template)

    def _create_qa_prompt(self):
        template = """
        你是一个智能助手，任务是根据 Neo4j Cypher 查询结果生成简洁、自然且易懂的回答。

如果查询结果为空：请礼貌地告知用户目前没有相关信息来回答他们的问题。
如果查询结果不为空：请清晰地回答用户的问题，并根据提供的信息生成一段简洁、易懂的自然语言回答。确保正确使用标点符号，保持语句流畅。同时，提醒用户：查询结果仅供参考，具体情况请咨询专业医生或直接就医。
注意：

请避免使用过于复杂的术语或专业语言，尽量使回答对普通用户友好且易于理解。
保持语气自然、友善，避免过于生硬或机械的表达。
查询结果：{context}
用户提问：{question}
        """
        return PromptTemplate(
            input_variables=["context", "question"], template=template
        )

    def _initialize_graph(self):
        try:
            graph = Neo4jGraph(
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
            )
            graph.refresh_schema()
            return graph
        except Exception as e:
            raise RuntimeError(f"无法连接到Neo4j数据库：{e}")

    def _initialize_chain(self):
        try:
            chain = GraphCypherQAChain.from_llm(
                cypher_llm=ChatNVIDIA(
                    model="meta/llama-3.1-70b-instruct",
                    api_key=self.nvidia_api_key,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024,
                ),
                qa_llm=ChatNVIDIA(
                    model="meta/llama-3.1-70b-instruct",
                    api_key=self.nvidia_api_key,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024,
                ),
                graph=self.graph,
                verbose=True,
                qa_prompt=self.qa_generation_prompt,
                cypher_prompt=self.cypher_generation_prompt,
                validate_cypher=True,
                top_k=100,
                allow_dangerous_requests=True,
            )
            return chain
        except Exception as e:
            raise RuntimeError(f"无法初始化GraphCypherQAChain：{e}")

    def query(self, question):
        try:
            response = self.hospital_cypher_chain.invoke(question)
            return response['result']
        except Exception as e:
            return f"查询过程中出现错误：{e}"


def run(question):
    neo4j_url = 'neo4j://localhost:7687'
    neo4j_username = "neo4j"
    neo4j_password = "Wszyd780"
    nvidia_api_key = (
        "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"
    )

    graph_chain = NL2Cyp(neo4j_url, neo4j_username, neo4j_password, nvidia_api_key)
    answer = graph_chain.query(question)
    return answer

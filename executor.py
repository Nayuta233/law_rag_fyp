import argparse
import logging
import sys
import re
import os
import argparse
import pdb

import requests
from pathlib import Path
from urllib.parse import urlparse

from llama_index import ServiceContext, StorageContext
from llama_index import set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.llms import OpenAI
from llama_index.readers.file.flat_reader import FlatReader
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser.text import SentenceWindowNodeParser
from llama_index.node_parser import SimpleNodeParser

from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole, PromptTemplate
from llama_index.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor import SentenceTransformerRerank
#from llama_index.indices import ZillizCloudPipelineIndex
from custom.zilliz.base import ZillizCloudPipelineIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import BaseNode, ImageNode, MetadataMode


from trulens_eval import Tru
from trulens_eval import Feedback,TruLlama
from trulens_eval import OpenAI as fOpenAI
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback import GroundTruthAgreement
import numpy as np
import nest_asyncio

from custom.law_sentence_window import LawSentenceWindowNodeParser
from custom.llms.QwenLLM import QwenUnofficial
from custom.llms.GeminiLLM import Gemini
from custom.llms.proxy_model import ProxyModel
from pymilvus import MilvusClient

QA_PROMPT_TMPL_STR = (
    "请你仔细阅读相关内容，结合法律资料进行回答。每一条法律资料使用'出处：《书名》原文内容'的形式标注 (如果回答请清晰无误地引用原文,先给出回答，再贴上对应的原文，使用《书名》用[]将原文括起来进行标识)，如果发现资料无法得到答案，就回答不知道 \n"
    "搜索的相关资料如下所示.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "问题: {query_str}\n"
    "答案: "
)
QA_PROMPT_TMPL_STR_ENG =("Please read the relevant content carefully and answer based on legal materials. Each legal reference should be marked in the form of 'Source: <Book Title> original text' (if answering, please quote the original text clearly and accurately, provide the answer first, and then paste the corresponding original text into <Book Title>[]). If the information cannot be found to answer, respond with 'don't know'.\n"
                         "The related information from the search is as follows:\n"
                         "---------------------\n"
                         "{context_str}\n"
                         "---------------------\n"
                         "query: {query_str}\n"
                         "answer: "

)

QA_SYSTEM_PROMPT = "你是一个严谨的法律知识问答智能体，你会仔细阅读法律材料并给出准确的回答,你的回答都会非常准确。因为你在回答的之后，使用在《书名》[]内给出原文用来支撑你回答的证据.并且你会在开头说明原文是否有回答所需的知识"
QA_SYSTEM_PROMPT_ENG = 'You are a meticulous legal knowledge question-answering AI. You will carefully read legal materials and provide accurate answers. Your responses are highly precise. After answering, you will paster the original text into "<Book Title>[]" to support your answer. You will also specify at the beginning whether the original text contains the required knowledge for the answer.'

REFINE_PROMPT_TMPL_STR = ( 
    "你是一个法律知识回答修正机器人，你严格按以下方式工作"
    "1.只有原答案为不知道时才进行修正,否则输出原答案的内容\n"
    "2.修正的时候为了体现你的精准和客观，你非常喜欢使用《书名》[]将原文展示出来.\n"
    "3.如果感到疑惑的时候，就用原答案的内容回答。"
    "新的知识: {context_msg}\n"
    "问题: {query_str}\n"
    "原答案: {existing_answer}\n"
    "新答案: "
)
REFINE_PROMPT_TMPL_STR_ENG = (
    "You are a legal knowledge correction robot, and you strictly follow the following rules:"
    "1.Only correct when the original answer is 'don't know'; otherwise, output the content of the original answer.\n"
    "2.When correcting, to demonstrate your precision and objectivity, you prefer displaying the original text by '<Book Title>[]'.\n"
    "3.If you feel unsure, respond with the content of the original answer."
    "New knowledge: {context_msg}\n"
    "Query: {query_str}\n"
    "Original answer: {existing_answer}\n"
    "New answer: "
)

EVAL_QA = [{"query":"小明今年15岁 他故意杀人是否会判处死刑？","response":"15岁杀人不会判死刑。根据《中华人民共和国刑法》第十七条和第四十九条的相关规定，已满十四周岁不满十六周岁的人，犯故意杀人、故意伤害致人重伤或者死亡、强奸、抢劫、贩卖毒品、放火、爆炸、投放危险物质罪的，应当负刑事责任，但犯罪的时候不满十八周岁的人，不适用死刑。已满十四周岁不满十八周岁的人犯罪，应当从轻或者减轻处罚。"},
                  {"query":"王某和郑某合谋骗取银行贷款，根据分工，郑某去伪造材料，之后郑某找到保险公司，让保险公司承保其从银行的“贷款”，但王某对于郑某骗取保险公司的行为不知情，二人骗取银行贷款600万后逃匿。银行最后找到保险公司要求赔偿，保险公司如约赔偿600万。郑某将面临哪些法律处罚", "response":"郑某可能构成贷款诈骗罪和保险诈骗罪"},
                  {"query":"赵某敲诈勒索周某10万元，以威胁网络上曝光其隐私为由，周某迫于隐私，给了10万，赵某指示周某把10万元现金置于指定的某垃圾箱内。赵某把真相告诉了刘某，让刘某去取钱，刘某前往把垃圾箱里的10万元拿走，并与赵某各分得5万元。刘某的行为构成什么罪","response":"刘某可能构成敲诈勒索罪或侵占罪"},
                  {"query":"2月13日晚，马加爵趁唐学李不备用石工锤砸向其头部，杀死唐学李后马用塑料袋扎住唐的头部，藏进衣柜锁好。14日晚，邵瑞杰回到宿舍，因隔壁宿舍同学已经回来他只好回到317室。马加爵趁其洗脚时用石工锤把他砸死。15日中午，杨开红到317室找马加爵打牌，碰巧马加爵正在处理尸体，马怕事情泄漏用石工锤杀死杨开红。当晚马加爵到龚博的宿舍，骗龚博说317室打牌三缺一，将其引到317室后用石工锤杀死。四人的尸体均被马加爵用黑色塑料袋扎住头部后放入衣柜锁住，15日当天，马加爵到云南省工商银行汇通支行学府路储蓄所分两次提取了350元和100元人民币现金。杀死4人后，马加爵在2月17日带着现金和自己之前制作的假身份证乘坐火车离开。马加爵将面临什么判罚","response":"马加爵无视国家法律，因不能正确处理人际关系，为琐事与同学积怨，即产生报复杀人的恶念，并经周密策划和准备，先后将四名同学残忍地杀害，主观上具有非法剥夺他人生命的故意，客观上实施了非法剥夺他人生命的行为，已构成故意杀人罪。在整个犯罪过程中，马加爵杀人犯意坚决，作案手段残忍；杀人后藏匿被害人尸体并畏罪潜逃，犯罪行为社会危害极大，情节特别恶劣，后果特别严重，应依法严惩。此外马加爵还涉嫌伪造，变卖居民身份证罪"}]

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_github_folder_url(url):
    return url.startswith('https://raw.githubusercontent.com/') and '.' not in os.path.basename(url)


def get_branch_head_sha(owner, repo, branch):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{branch}"
    response = requests.get(url)
    data = response.json()
    sha = data['object']['sha']
    return sha

def get_github_repo_contents(repo_url):
    # repo_url example: https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/
    repo_owner = repo_url.split('/')[3]
    repo_name = repo_url.split('/')[4]
    branch = repo_url.split('/')[5]
    folder_path = '/'.join(repo_url.split('/')[6:])
    sha = get_branch_head_sha(repo_owner, repo_name, branch)
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{sha}?recursive=1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            raw_urls = []
            for file in data['tree']:
                if file['path'].startswith(folder_path) and file['path'].endswith('.txt'):
                    raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
                    raw_urls.append(raw_url)
            return raw_urls
        else:
            print(f"Failed to fetch contents. Status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to fetch contents. Error: {str(e)}")
    return []

class Executor:
    def __init__(self, model):
        pass

    def build_index(self, path, overwrite):
        pass

    def build_query_engine(self):
        pass
     
    def delete_file(self, path):
        pass
    
    def query(self, question):
        pass

    def eval(self):
        pass
 

class MilvusExecutor(Executor):
    def __init__(self, config):
        self.index = None
        self.query_engine = None
        self.config = config
        self.node_parser = LawSentenceWindowNodeParser.from_defaults(
            sentence_splitter=lambda text: re.findall("[^,.;。？！]+[,.;。？！]?", text),
            window_size=config.milvus.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",)

        embed_model = HuggingFaceEmbedding(model_name=config.embedding.name)

        # 使用Qwen 通义千问模型
        if config.llm.name.find("qwen") != -1:
            llm = QwenUnofficial(temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)
        elif config.llm.name.find("gemini") != -1:
            llm = Gemini(temperature=config.llm.temperature, model_name=config.llm.name, max_tokens=2048)
        elif 'proxy_model' in config.llm:
            llm = ProxyModel(model_name=config.llm.name, api_base=config.llm.api_base, api_key=config.llm.api_key,
                             temperature=config.llm.temperature,  max_tokens=2048)
            print(f"使用{config.llm.name},PROXY_SERVER_URL为{config.llm.api_base},PROXY_API_KEY为{config.llm.api_key}")
        else:
            api_base = None
            if 'api_base' in config.llm:
                api_base = config.llm.api_base
            llm = OpenAI(api_base = api_base, temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        set_global_service_context(service_context)
        rerank_k = config.milvus.rerank_topk
        self.rerank_postprocessor = SentenceTransformerRerank(
            model=config.rerank.name, top_n=rerank_k)
        self._milvus_client = None
        self._debug = False
        
    def set_debug(self, mode):
        self._debug = mode

    def build_index(self, path, overwrite):
        config = self.config
        vector_store = MilvusVectorStore(
            uri = f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name = config.milvus.collection_name,
            overwrite=overwrite,
            dim=config.embedding.dim)
        self._milvus_client = vector_store.milvusclient
         
        if path.endswith('.txt'):
            if os.path.exists(path) is False:
                print(f'(rag) cannot find document:{path}')
                return
            else:
                documents = FlatReader().load_data(Path(path))
                documents[0].metadata['file_name'] = documents[0].metadata['filename'] 
        elif os.path.isfile(path):           
            print('(rag) only txt document supported')
        elif os.path.isdir(path):
            if os.path.exists(path) is False:
                print(f'(rag) cannot find path: {path}')
                return
            else:
                documents = SimpleDirectoryReader(path).load_data()
        else:
            return

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

    def _get_index(self):
        config = self.config
        vector_store = MilvusVectorStore(
            uri = f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name = config.milvus.collection_name,
            dim=config.embedding.dim)
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self._milvus_client = vector_store.milvusclient

    def build_query_engine(self):
        config = self.config
        lan = config.lan
        if self.index is None:
            self._get_index()
        self.query_engine = self.index.as_query_engine(node_postprocessors=[
            self.rerank_postprocessor,
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ])
        self.query_engine._retriever.similarity_top_k=config.milvus.retrieve_topk
        if lan == 'Chinese':
            message_templates = [
                ChatMessage(content=QA_SYSTEM_PROMPT, role=MessageRole.SYSTEM),
                ChatMessage(
                    content=QA_PROMPT_TMPL_STR,
                    role=MessageRole.USER,
                ),
            ]
        else:
            message_templates = [
                ChatMessage(content=QA_SYSTEM_PROMPT_ENG, role=MessageRole.SYSTEM),
                ChatMessage(
                    content=QA_PROMPT_TMPL_STR_ENG,
                    role=MessageRole.USER,
                ),
            ]           
        chat_template = ChatPromptTemplate(message_templates=message_templates)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": chat_template}
        )
        if lan == 'Chinese':
            self.query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = REFINE_PROMPT_TMPL_STR
        else:
            self.query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = REFINE_PROMPT_TMPL_STR_ENG
    def delete_file(self, path):
        config = self.config
        if self._milvus_client is None:
            self._get_index()
        num_entities_prev = self._milvus_client.query(collection_name=config.milvus.collection_name,filter="",output_fields=["count(*)"])[0]["count(*)"]
        res = self._milvus_client.delete(collection_name=config.milvus.collection_name, filter=f"file_name=='{path}'")
        num_entities = self._milvus_client.query(collection_name=config.milvus.collection_name,filter="",output_fields=["count(*)"])[0]["count(*)"]
        print(f'(rag) {num_entities} context(s) exist, remove {num_entities_prev - num_entities} context(s)')
    
    def query(self, question):
        if self.index is None:
            self._get_index()
        if question.endswith('?') or question.endswith('？'):
            question = question[:-1]
        if self._debug is True:
            contexts = self.query_engine.retrieve(QueryBundle(question))
            for i, context in enumerate(contexts): 
                print(f'{question}', i)
                content = context.node.get_content(metadata_mode=MetadataMode.LLM)
                print(content)
            print('-------------------------------------------------------参考资料---------------------------------------------------------')
        response = self.query_engine.query(question)
        return response
    def eval(self):                
        nest_asyncio.apply()
        #创建评估器对象
        tru = Tru()
        #定义评估记录器
        def get_prebuilt_trulens_recorder(query_engine, app_id):
            openai = fOpenAI()
            qa_relevance = (
                Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
                .on_input_output()
            )
            qs_relevance = (
                Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
                .on_input()
                .on(TruLlama.select_source_nodes().node.text)
                .aggregate(np.mean)
            )
            grounded = Groundedness(groundedness_provider=openai)
            groundedness = (
                Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
                    .on(TruLlama.select_source_nodes().node.text)
                    .on_output()
                    .aggregate(grounded.grounded_statements_aggregator)
            )
            # Define a groundtruth feedback function
            groundtruth = (
                Feedback(GroundTruthAgreement(EVAL_QA,provider = openai).agreement_measure, name = "Ground Truth").on_input_output()
            )
            feedbacks = [qa_relevance, qs_relevance, groundedness,groundtruth]
            tru_recorder = TruLlama(
                query_engine,
                app_id=app_id,
                feedbacks=feedbacks
            )
            return tru_recorder 
        #执行评估
        def run_evals(tru_recorder, query_engine):
            for qa in EVAL_QA:
                with tru_recorder as recording:
                    response = query_engine.query(qa['query'])
        #初始化评估数据库
        Tru().reset_database()
        tru_recorder_1 = get_prebuilt_trulens_recorder(
            self.query_engine,
            app_id='sentence window engine 1')
        run_evals(tru_recorder_1, self.query_engine)
        Tru().run_dashboard()



class PipelineExecutor(Executor):
    def __init__(self, config):
        self.ZILLIZ_CLUSTER_ID = os.getenv("ZILLIZ_CLUSTER_ID")
        self.ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
        self.ZILLIZ_PROJECT_ID = os.getenv("ZILLIZ_PROJECT_ID") 
        self.ZILLIZ_CLUSTER_ENDPOINT = f"https://{self.ZILLIZ_CLUSTER_ID}.api.gcp-us-west1.zillizcloud.com"
    
        self.config = config
        if len(self.ZILLIZ_CLUSTER_ID) == 0:
            print('ZILLIZ_CLUSTER_ID 参数为空')
            exit()

        if len(self.ZILLIZ_TOKEN) == 0:
            print('ZILLIZ_TOKEN 参数为空')
            exit()
        
        self.config = config
        self._debug = False

        if config.llm.name.find("qwen") != -1:
            llm = QwenUnofficial(temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)
        elif config.llm.name.find("gemini") != -1:
            llm = Gemini(model_name=config.llm.name, temperature=config.llm.temperature, max_tokens=2048)
        else:
            api_base = None
            if 'api_base' in config.llm:
                api_base = config.llm.api_base
            llm = OpenAI(api_base = api_base, temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=None)
        self.service_context = service_context
        set_global_service_context(service_context)
        self._initialize_pipeline(service_context)

        #rerank_k = config.rerankl
        #self.rerank_postprocessor = SentenceTransformerRerank(
        #    model="BAAI/bge-reranker-large", top_n=rerank_k)

    def set_debug(self, mode):
        self._debug = mode

    def _initialize_pipeline(self, service_context: ServiceContext):
        config = self.config
        try:
            self.index = ZillizCloudPipelineIndex(
                project_id = self.ZILLIZ_PROJECT_ID,
                cluster_id=self.ZILLIZ_CLUSTER_ID,
                token=self.ZILLIZ_TOKEN,
                collection_name=config.pipeline.collection_name,
                service_context=service_context,
             )
            if len(self._list_pipeline_ids()) == 0:
                self.index.create_pipelines(
                    metadata_schema={"digest_from":"VarChar"}, chunk_size=self.config.pipeline.chunk_size
                )
        except Exception as e:
            print('(rag) zilliz pipeline 连接异常', str(e))
            exit()
        try:
            self._milvus_client = MilvusClient(
                uri=self.ZILLIZ_CLUSTER_ENDPOINT, 
                token=self.ZILLIZ_TOKEN 
            )
        except Exception as e:
            print('(rag) zilliz cloud 连接异常', str(e))

    def build_index(self, path, overwrite):
        config = self.config
        if not is_valid_url(path) or 'github' not in path:
            print('(rag) 不是一个合法的url，请尝试`https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/baihuasanguozhi.txt`')
            return
        if overwrite == True:
            self._milvus_client.drop_collection(config.pipeline.collection_name)
            pipeline_ids = self._list_pipeline_ids()
            self._delete_pipeline_ids(pipeline_ids)

            self._initialize_pipeline(self.service_context)

        if is_github_folder_url(path):
            urls = get_github_repo_contents(path)
            for url in urls:
                print(f'(rag) 正在构建索引 {url}')
                self.build_index(url, False)  # already deleted original collection
        elif path.endswith('.txt'):
            self.index.insert_doc_url(
                url=path,
                metadata={"digest_from": LawSentenceWindowNodeParser.book_name(os.path.basename(path))},
            )
        else:
            print('(rag) 只有github上以txt结尾或文件夹可以被支持。')

    def build_query_engine(self):
        config = self.config
        self.query_engine = self.index.as_query_engine(
          search_top_k=config.pipeline.retrieve_topk)
        message_templates = [
            ChatMessage(content=QA_SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=QA_PROMPT_TMPL_STR,
                role=MessageRole.USER,
            ),
        ]
        chat_template = ChatPromptTemplate(message_templates=message_templates)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": chat_template}
        )
        self.query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = REFINE_PROMPT_TMPL_STR


    def delete_file(self, path):
        config = self.config
        if self._milvus_client is None:
            self._get_index()
        num_entities_prev = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        res = self._milvus_client.delete(collection_name=config.milvus.collection_name, filter=f"doc_name=='{path}'")
        num_entities = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        print(f'(rag) 现有{num_entities}条，删除{num_entities_prev - num_entities}条数据')

    def query(self, question):
        if self.index is None:
            self.get_index()
        if question.endswith("?") or question.endswith("？"):
            question = question[:-1]
        if self._debug is True:
            contexts = self.query_engine.retrieve(QueryBundle(question))
            for i, context in enumerate(contexts): 
                print(f'{question}', i)
                content = context.node.get_content(metadata_mode=MetadataMode.LLM)
                print(content)
            print('-------------------------------------------------------参考资料---------------------------------------------------------')
        response = self.query_engine.query(question)
        return response

    def _list_pipeline_ids(self):
        url = f"https://controller.api.gcp-us-west1.zillizcloud.com/v1/pipelines?projectId={self.ZILLIZ_PROJECT_ID}"
        headers = {
            "Authorization": f"Bearer {self.ZILLIZ_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        collection_name = self.config.milvus.collection_name
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        pipeline_ids = []
        for pipeline in response_dict['data']: 
            if collection_name in  pipeline['name']:
                pipeline_ids.append(pipeline['pipelineId'])
            
        return pipeline_ids

    def _delete_pipeline_ids(self, pipeline_ids):
        for pipeline_id in pipeline_ids:
            url = f"https://controller.api.gcp-us-west1.zillizcloud.com/v1/pipelines/{pipeline_id}/"
            headers = {
                "Authorization": f"Bearer {self.ZILLIZ_TOKEN}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            response = requests.delete(url, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(response.text)


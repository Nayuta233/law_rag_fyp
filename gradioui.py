from executor import MilvusExecutor
from executor import PipelineExecutor

import gradio as gr

from cli import CommandLine, read_yaml_config  # 导入 CommandLine 类

resolutions = ["milvus", "pipeline"]
languages = ["Chinese", "English"]
laws = ["data/Chinese_law/criminal_specific_provisions.txt",
        "data/Chinese_law/criminal_general_provisions.txt",
        "data/Chinese_law/civil_code_contracts.txt",
        "data/Chinese_law/civil_code_general_provisions.txt",
        "data/Chinese_law/civil_code_marriage_and_family.txt",
        "data/Chinese_law/civil_code_personality_rights.txt",
        "data/Chinese_law/civil_code_real_rights.txt",
        "data/Chinese_law/civil_code_succession.txt",
        "data/Chinese_law/civil_code_tort_liability.txt",
        "data/Singaporean_law/Penal_Code_1871.txt"
        ]

build_tasks = ["Build up context", "remove context"]
query_tasks = ["ask", "ask+return retrieved context"]


class GradioCommandLine(CommandLine):
    def __init__(self, cfg, cfg_eng):
        super().__init__(cfg, cfg_eng)
        self.config_path = cfg
        self.config_eng_path = cfg_eng
    def index(self, task, path, overwrite):
        if task == "build up context":
            self._executor.build_index(path, overwrite)
            return "context built up"
        elif task == "remove context":
            self._executor.delete_file(path)
            return "context removed"

    def query(self, task, question):
        if task == "ask":
            return self._executor.query(question)
        elif task == "ask+return retrieved context":
            self._executor.set_debug(True)
            return self._executor.query(question)


def initialize_cli(cfg_path,cfg_eng_path, resolution, language):
    global cli_instance
    cli_instance = GradioCommandLine(cfg_path,cfg_eng_path)
    if language == "Chinese":
        conf = read_yaml_config(cli_instance.config_path)
    else:
        conf = read_yaml_config(cli_instance.config_eng_path)
    if resolution == "milvus":
        cli_instance._executor = MilvusExecutor(conf)
        cli_instance._mode = "milvus"
    else:
        cli_instance._executor = PipelineExecutor(conf)
        cli_instance._mode = "pipeline"
    cli_instance._executor.build_query_engine()
    return "CLI initilized"


with gr.Blocks() as demo:
    # 初始化
    gr.Interface(fn=initialize_cli,
                 inputs=[gr.Textbox(
                     lines=1, value="cfgs/config.yaml"),
                     gr.Textbox(
                     lines=1, value="cfgs/config_eng.yaml"),
                     gr.Dropdown(resolutions, label="category", value="milvus"),
                     gr.Dropdown(languages, label="language", value="Chinese")],
                 outputs="text",
                 submit_btn="initilize", clear_btn="clear")
    # Build up context
    gr.Interface(fn=lambda command, argument, overwrite: cli_instance.index(command, argument, overwrite),
                 inputs=[gr.Dropdown(choices=build_tasks, label="command selection", value="Build up context"),
                         gr.Dropdown(laws, label="laws", value="data/Chinese_law/criminal_specific_provisions.txt"), gr.Checkbox(label="cover previous context")], outputs="text",
                 submit_btn="submit", clear_btn="clear")

    # 提问
    gr.Interface(fn=lambda command, argument: cli_instance.query(command, argument),
                 inputs=[gr.Dropdown(choices=query_tasks, label="command selection", value="ask"),
                         gr.Textbox(label="ask")], outputs="text",
                 submit_btn="submit", clear_btn="clear")
    with open("docs/web_ui.md", "r", encoding="utf-8") as f:
        article = f.read()
    gr.Markdown(article)

if __name__ == '__main__':
    # 启动 Gradio 界面
    demo.launch()

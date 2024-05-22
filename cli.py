from executor import MilvusExecutor
from executor import PipelineExecutor

import yaml
from easydict import EasyDict
import argparse

def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)

class CommandLine():
    def __init__(self, config_path, config_eng_path):
        self._mode = None
        self._executor = None
        self.config_path = config_path
        self.config_eng_path = config_eng_path

    def show_start_info(self):
        with open('./start_info.txt') as fw:
            print(fw.read())

    def run(self):
        self.show_start_info()
        while True:
            print('Chinese or English?')
            lan = input('(rag) ')
            if lan == 'Chinese':
                conf = read_yaml_config(self.config_path)
            else:
                conf = read_yaml_config(self.config_eng_path)
            print('(rag) choose [milvus|pipeline] mode')
            mode = input('(rag) ')
            if mode == 'milvus':
                self._executor = MilvusExecutor(conf) 
                print('(rag) milvus mode has been chosen')
                print('  1.type `build data/Chinese_law/criminal_general_provisions.txt` to build up knowledge base.')
                print('  2.type `ask` to ask query based on exisiting context, `-d` for debug mode.')
                print('  3.type`remove criminal_general_provisions.txt` to remove existing context.')
                self._mode = 'milvus'
                break
            elif mode == 'pipeline':
                self._executor = PipelineExecutor(conf)
                print('(rag) pipeline has been chosen, type `build https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/baihuasanguozhi.txt` to build knowledge base.')
                print('  1.type`build https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/baihuasanguozhi.txt`to build knowledge base.')
                print('  2.type `ask` to ask query based on exisiting context, `-d` for debug mode.')
                print('  3.type`remove criminal_general_provisions.txt` to remove existing context.')
                self._mode = 'pipeline'
                break
            elif mode == 'quit':
                self._exit()
                break
            else:
                print(f'(rag) {mode} is not known mode, choose [milvus|pipeline]mode,or type "quit".')
        assert self._mode != None
        while True:
            command_text = input("(rag) ")
            self.parse_input(command_text)

    def parse_input(self, text):
        commands = text.split(' ')
        if commands[0] == 'build':
            if len(commands) == 3:
                if commands[1] == '-overwrite':  
                    print(commands)
                    self.build_index(path=commands[2], overwrite=True)
                else:
                    print('(rag) build only support `-overwrite`')
            elif len(commands) == 2:
                self.build_index(path=commands[1], overwrite=False)
        elif commands[0] == 'ask':
            if len(commands) == 2:
                if commands[1] == '-d':
                    self._executor.set_debug(True)
                else: 
                    print('(rag) ask only support `-d` ')
            else:
                self._executor.set_debug(False)
            self.question_answer()
        elif commands[0] == 'remove':
            if len(commands) != 2:
                print('(rag) remove only accept one parameter.')
            self._executor.delete_file(commands[1])
            
        elif 'quit' in commands[0]:
            self._exit()

        elif commands[0] == 'eval':
            self.question_eval()

        else: 
            print('(rag) only [build|ask|remove|eval|quit] provided, please try again.')
            
    def query(self, question):
        ans = self._executor.query(question)
        print(ans)
        print('+---------------------------------------------------------------------------------------------------------------------+')
        print('\n')
    
    def eval(self):
        self._executor.eval()

    def build_index(self, path, overwrite):
        self._executor.build_index(path, overwrite)
        print('(rag) build up of context finished')

    def remove(self, filename):
        self._executor.delete_file(filename)
        
    def question_answer(self):
        self._executor.build_query_engine()
        while True: 
            question = input("(rag) query: ")
            if question == 'quit':
                print('(rag) quit ask mode')
                break
            elif question == "":
                continue
            else:
                pass
            self.query(question)
    def question_eval(self):
        self._executor.build_query_engine()
        self.eval()

    def _exit(self):
        exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to the configuration file', default='cfgs/config.yaml')
    parser.add_argument('--cfg_eng', type=str, help='Path to the configuration file', default='cfgs/config_eng.yaml')
    args = parser.parse_args()

    cli = CommandLine(args.cfg,args.cfg_eng)
    cli.run()


import argparse

def add_model_config_args(parser):
    """Model arguments"""
    parser.add_argument("-w", "--webglm_ckpt_path", type=str, default=None, help="path to the webglm checkpoint, default to $WEBGLM_CKPT or THUDM/WebGLM")
    
    parser.add_argument("-r", "--retriever_ckpt_path", type=str, default=None, help="path to the retriever checkpoint, default to $WEBGLM_RETRIEVER_CKPT")
    
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device to run the model, default to cuda")
    
    parser.add_argument("-b", "--filter_max_batch_size", type=int, default=50, help="max batch size for the retriever, default to 50")
    
    parser.add_argument("-s", "--serpapi_key", type=str, default=None, help="serpapi key for the searcher, default to $SERPAPI_KEY")
    parser.add_argument("--searcher", type=str, default="serpapi", help="searcher to use (serpapi or bing), default to serpapi")
    
    return parser

def add_evaluation_args(parser):
    """Evaluation arguments"""
    parser.add_argument("-t", "--task", type=str, default=None, help="evaluate task, choose from nq_open, web_questions, triviaqa")
    
    parser.add_argument("-p", "--evaluate_task_data_path", type=str, default=None, help="data path of the evaluate task")
    
    return parser

def get_args(args_list=None, parser=None):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description='webglm')
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    
    parser = add_model_config_args(parser)
    parser = add_evaluation_args(parser)
    
    return parser.parse_args()
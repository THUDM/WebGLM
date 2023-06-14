from arguments import get_args
from model import load_model

def main():
    args = get_args()
    
    webglm = load_model(args)
    
    task = args.task
    if task == 'triviaqa':
        from evaluate.triviaqa import eval
    elif task == 'nq_open':
        from evaluate.eval import eval
    elif task == 'web_questions':
        from evaluate.eval import eval
    else:
        raise "Task Name Error!"
    
    print('WebGLM Initialize Done. Start Evaluating...')
    result = eval(webglm, args)
    print(f'Result: {result}')
    print('Evaluate Done')

if __name__ == "__main__":
    main()
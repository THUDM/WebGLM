from model import load_model, citation_correction
import argparse
from arguments import add_model_config_args

if __name__ == '__main__':
    
    arg = argparse.ArgumentParser()
    add_model_config_args(arg)
    args = arg.parse_args()
    
    webglm = load_model(args)
    
    while True:
        question = input("[Enter to Exit] >>> ")
        question = question.strip()
        if not question:
            break
        if question == "quit":
            break
        final_results = {}
        for results in webglm.stream_query(question):
            final_results.update(results)
            if "references" in results:
                for ix, ref in enumerate(results["references"]):
                    print("Reference [%d](%s): %s"%(ix + 1, ref['url'], ref['text']))
            if "answer" in results:
                print("\n%s\n"%citation_correction(results["answer"], [ref['text'] for ref in final_results["references"]]))
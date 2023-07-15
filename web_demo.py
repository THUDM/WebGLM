import gradio as gr
from model import citation_correction, load_model
import argparse

from arguments import add_model_config_args

TOTAL_NUM = 10
CSS = """
    #col {
        width: min(100%, 800px);
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        margin: auto;
    }
    
    footer{display:none !important}
"""


    
# a summary structure ( use <summary> tag in html )
# title is in summary, click to expand
# in the container, there is an icon that can be clicked to jump to url.
# the other part is the text.
ref_html = """

<details style="border: 1px solid #ccc; padding: 10px; border-radius: 4px; margin-bottom: 4px">
    <summary style="display: flex; align-items: center; font-weight: bold;">
        <span style="margin-right: 10px;">[{index}] {title}</span>
        <a href="{url}" style="text-decoration: none; background: none !important;" target="_blank">
            <!--[Here should be a link icon]-->
            <i style="border: solid #000; border-width: 0 2px 2px 0; display: inline-block; padding: 3px; transform:rotate(-45deg); -webkit-transform(-45deg)"></i>   
        </a>
    </summary>
    <p style="margin-top: 10px;">{text}</p>
</details>

"""

def query(query: str):
    """ Query the model """
    
    refs = []
    answer = "Loading ..."
    
    yield answer, ""
    
    for resp in webglm.stream_query(query):
        if "references" in resp:
            refs = resp["references"]
        if "answer" in resp:
            answer = resp["answer"]
            answer = citation_correction(answer, [ref['text'] for ref in refs])
        yield answer, "<h3>References (Click to Expand)</h3>" + "\n".join([ref_html.format(**item, index = idx + 1) for idx, item in enumerate(refs)])
    
if __name__ == '__main__':
    
    arg = argparse.ArgumentParser()
    add_model_config_args(arg)
    args = arg.parse_args()
    
    webglm = load_model(args)
    
    with gr.Blocks(theme=gr.themes.Base(), css=CSS) as demo:
        
        with gr.Column(elem_id='col'):
            gr.Markdown(
            """
            # WebGLM Demo
            """)
            with gr.Row():
                # with gr.Column(scale=8):
                query_box = gr.Textbox(show_label=False, placeholder="Enter question and press ENTER").style(container=False)
                # with gr.Column(scale=1, min_width=60):
                #     query_button = gr.Button('Query')
            
            answer_box = gr.Textbox(show_label=False, value='', lines=5)
            
            # with gr.Box():
            ref_boxes = gr.HTML(label="References")
    
            # with gr.Column() as refs_col:
            #     ref_boxes = []
            #     for i in range(TOTAL_NUM):
            #         ref_boxes.append(gr.Textbox(f"Textbox {i}", visible=False)) 
 
        query_box.submit(query, query_box, [answer_box, ref_boxes])
        # query_button.click(query, query_box, [answer_box, ref_boxes])

    demo.queue()
    demo.launch()
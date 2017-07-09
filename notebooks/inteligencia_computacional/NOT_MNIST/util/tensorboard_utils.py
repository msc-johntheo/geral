from keras.callbacks import TensorBoard
from IPython.display import clear_output, Image, display, HTML

"""
CALLBACK
"""
def make_tb_callback(run):
    """
    Make a callback function to be called during training.
    
    Args:
        run: folder name to save log in. 
    
    Made this a function since we need to recreate it when
    resetting the session. 
    (See https://github.com/fchollet/keras/issues/4499)
    """
    return TensorBoard(
            # where to save log file
            log_dir='./graph-tensorboard/' + run,
            # how often (in epochs) to compute activation histograms
            # (more frequently slows down training)
            histogram_freq=1, 
            # whether to visualize the network graph.
            # This now works reasonably in Keras 2.01!
            write_graph=True,
            # if true, write layer weights as images
            write_images=False)

"""
VIZUALIZATION
(See https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter/38192374#38192374)
"""
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
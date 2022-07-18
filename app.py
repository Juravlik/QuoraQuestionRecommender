from flask import Flask, request, jsonify
from app_utils.app_handler import Handler
import json

app = Flask(__name__)

with open('configs/app_config.json') as json_file:
    config = json.load(json_file)


@app.route('/ping')
def ping():
    if not handler.model_is_ready:
        return jsonify(status="not ready")
    return jsonify(status="ok")


@app.route('/query', methods=['POST'])
def query():
    if not handler.model_is_ready or not handler.index_is_ready:
        return json.dumps({"status": "FAISS is not initialized!"})

    queries = json.loads(request.json)['queries']
    lang_check, suggestions = handler.similarity_search(queries)

    return jsonify({'lang_check': lang_check, 'suggestions': suggestions})


@app.route('/update_index', methods=['POST'])
def update_index():
    documents = json.loads(request.json)['documents']
    handler.build_index(documents)
    index_size = handler.index_size

    return jsonify(status="ok", index_size=index_size)


handler = Handler(config['emb_path_knrm'], config['mlp_path'], config['vocab_path'])
app.run(debug=False, port=config['port'], host=config['host'])

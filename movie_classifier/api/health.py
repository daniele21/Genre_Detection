from flask import make_response, jsonify
from flask import current_app as app


@app.route('/health', methods=['GET'])
def health():

    return make_response(jsonify({'Status': 'OK'}))

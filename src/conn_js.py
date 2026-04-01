from flask import Flask, jsonify, request
from flask_cors import CORS
import Graphrag


app = Flask(__name__, static_folder="../static/pages")
CORS(app, origins="http://localhost:3000")


# 用于处理聊天消息的 POST 路由
@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')  # 获取前端发送的消息
    bot_response = Graphrag.query_hospital_data(user_message)
    return jsonify({"response": bot_response})


if __name__ == '__main__':
    app.run(debug=True)

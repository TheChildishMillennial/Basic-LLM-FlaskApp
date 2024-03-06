from flask_app.app.llm import bp
from flask import request, jsonify
from flask_app.app.llm.basic_langchain_1 import basic_1
from flask_app.app.llm.basic_langchain_2 import basic_2
from flask_app.app.llm.basic_langchain_3 import basic_3
from flask_app.app.llm.memory_langchain_1 import memory_1
from flask_app.app.llm.memory_langchain_2 import memory_2
from flask_app.app.llm.memory_langchain_3 import memory_3
from flask_app.app.llm.memory_langchain_4 import memory_4
from flask_app.app.llm.memory_langchain_5 import memory_5
from flask_app.app.llm.summary_langchain_1 import summary_1
from flask_app.app.llm.summary_langchain_2 import summary_2
from flask_app.app.llm.agent_langchain_1 import agent_1, prompt


@bp.route('/basic-1', methods=['GET', 'POST'])
def basic1():
    user_input = request.json['message']

    response = basic_1.run(input=user_input)
    print(response)
    return jsonify({"message": response})


@bp.route('/basic-2', methods=['GET', 'POST'])
def basic2():
    user_input = request.json['message']

    response = basic_2.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/basic-3', methods=['GET', 'POST'])
def basic3():
    user_input = request.json['message']

    response = basic_3.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/memory-1', methods=['GET', 'POST'])
def memory1():
    user_input = request.json['message']

    response = memory_1.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/memory-2', methods=['GET', 'POST'])
def memory2():
    user_input = request.json['message']

    response = memory_2.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/memory-3', methods=['GET', 'POST'])
def memory3():
    user_input = request.json['message']

    response = memory_3.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/memory-4', methods=['GET', 'POST'])
def memory4():
    user_input = request.json['message']

    response = memory_4.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/memory-5', methods=['GET', 'POST'])
def memory5():
    user_input = request.json['message']

    response = memory_5.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/summary-1', methods=['GET', 'POST'])
def summary1():
    user_input = request.json['message']

    response = summary_1.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/summary-2', methods=['GET', 'POST'])
def summary2():
    user_input = request.json['message']

    response = summary_2.invoke({"query": user_input})
    print(response)
    return jsonify({"message": response})


@bp.route('/agent-1', methods=['GET', 'POST'])
def agent1():
    user_input = request.json['message']

    response = agent_1.invoke({"input": user_input})
    print(response)
    return jsonify({"message": response})
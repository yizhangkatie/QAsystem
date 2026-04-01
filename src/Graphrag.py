import NERmodel
import NL2Cyp
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def query_hospital_data(question):
    try:
        # 使用 NER 模型提取实体
        # entities = NERmodel.run(question)
        # logging.info(f"提取的实体: {entities}")

        # augmented_question = f"{question} 提到的实体包括: {', '.join(entities)}"
        augmented_question = question
        response = NL2Cyp.run(augmented_question)
        return response

    except Exception as e:
        return f"查询失败，请稍后重试或检查输入的问题格式。错误详情：{e}"


if __name__ == "__main__":
    while True:
        qur = input()
        print(query_hospital_data(qur))

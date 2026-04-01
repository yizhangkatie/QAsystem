import React from "react"
import LeftBotton from "../../components/left_button"
import TopCol from "../../components/top_col"
export default function Help() {
    return (
        <div className="relative flex h-screen w-full justify-center">
            <LeftBotton />
            <TopCol col_name={"使用指南"} />
            <div className="absolute top-12 flex pt-12 pl-16 w-full justify-center ">
                <div className="relative flex flex-col space-y-6 text-base">
                    <p>
                        欢迎使用中药领域的智能助手！以下是使用指南，让您轻松与中药问答机器人互动：
                    </p>

                    <p>
                        提问方式
                        <br />
                        您可以直接输入问题，如：
                        <br />
                        “哪种中药可以帮助改善睡眠？”
                        <br />
                        “缓解焦虑的中药有哪些？”
                    </p>

                    <p>
                        健康建议
                        <br />
                        系统可以提供与中药相关的健康建议、调理方案和中药材介绍，帮助您更好地理解中药的功效与用法。
                    </p>

                    <p>
                        关怀模式
                        <br />
                        支持更大的字体，更简洁易用的页面。
                    </p>

                    <p>
                        账户功能
                        <br />
                        登录后，您可以查看过往问题记录，方便复习与跟踪。账号还支持个性化设置，定制您的体验。
                    </p>

                    <p>
                        快来提问，开始您的中药养生之旅吧！
                        <br />
                    </p>

                    <p className="text-mypurple200">
                        提示：系统的回答仅供参考，复杂的健康问题请咨询专业医生
                        <br />
                    </p>
                    <button
                        onClick={() => (window.location.href = "/post/chat")}
                        className="absolute bottom-0 right-0 bg-mypurple100 rounded-lg w-24 h-8"
                    >
                        去互动
                    </button>
                </div>
            </div>
        </div>
    )
}

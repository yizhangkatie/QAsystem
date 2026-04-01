import React, { useState, useRef, useEffect } from "react"
import LeftBotton from "../../components/left_button"
import TopCol from "../../components/top_col"
import TypingEffectMessage from "../../components/typing"

const Chat = () => {
    const [isCareMode, setIsCareMode] = useState(false)
    const switchmode = () => {
        setIsCareMode(!isCareMode)
    }

    const [inputData, setInputData] = useState("")
    const [messages, setMessages] = useState([])
    const messageEndRef = useRef(null)

    const handleInputChange = (e) => {
        setInputData(e.target.value)
    }

const handleSubmit = () => {
    // 调用后端 API
    const newMessage = {
        type: "input",
        text: inputData
    }
    const input = newMessage.text
    console.log("Sending message:", inputData);
    setMessages((prevMessages) => [
        ...prevMessages,
        newMessage,
        {
            type: "output",
            text: ""
        }
    ]);
    setInputData("")
    fetch("http://127.0.0.1:5000/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message: input
            }),
        })
        .then((response) => {
            console.log("Response received:", response);

            return response.json()
        })
        .then((data) => {
            const outputMessage = {
                type: "output",
                text: data.response
            }
            console.log("dataResonse:", data.response)
            setMessages((prevMessages) => {
                const updatedMessages = [...prevMessages];
                updatedMessages[updatedMessages.length - 1] = outputMessage; // 替换最后一个空的 output 消息
                return updatedMessages;
            });
        })
        .catch((error) => {
            console.error("Error:", error)
        })
}
    useEffect(() => {
        messageEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages])
    return (
        <div className="flex h-screen w-full">
            <LeftBotton />
            <div className="relative flex flex-col w-48 h-full bg-gray-100 items-center">
                <div className="w-full h-12 bg-gray-100 border border-gray-200 flex items-center justify-center">
                    <p className="text-black text-base ">中药问答机器人</p>
                </div>
                <button
                    onClick={switchmode}
                    className="absolute bottom-12 w-36 h-12 bg-mypurple100 rounded-lg flex items-center justify-center"
                >
                    {isCareMode ? (
                        <p className="text-base">切换普通模式</p>
                    ) : (
                        <p className="text-base">切换关怀模式</p>
                    )}
                </button>
            </div>

            <div className="relative flex flex-1 justify-center">
                {isCareMode ? (
                    <TopCol col_name={"关怀模式"} />
                ) : (
                    <TopCol col_name={"普通模式"} />
                )}
                <div className="absolute top-0 w-full h-full flex flex-col">
                    <form
                        onSubmit={handleSubmit}
                        className="relative flex flex-col w-full h-full pt-12 items-center justify-center"
                    >
                        {/* 消息显示区域 */}
                        <div className="flex-1 overflow-y-auto pr-40 pb-16 w-full flex flex-col items-end">
                            {messages.map((message, index) => (
                                <div
                                    key={index}
                                    className={`w-4/5 text-black text-justify ${message.type === "input" ? "max-w-md w-auto bg-gray-100 rounded-xl p-4" : "max-w-2xl "} my-4 ${isCareMode? "text-lg":"text-sm"}`}
                                >
                                    {message.type === "input" ? (
                                        <p className="break-words whitespace-normal">
                                            {message.text}
                                        </p>
                                    ) : (
                                        <TypingEffectMessage
                                            text={message.text}
                                        />
                                    )}
                                </div>
                            ))}
                            <div ref={messageEndRef} />
                        </div>

                        {/* 输入框和按钮 */}
                        <div className="absolute bottom-0 flex flex-col w-2/3 bg-white h-16 items-center rounded-xl p-2">
                            <input
                                type="text"
                                value={inputData}
                                onChange={handleInputChange}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter") {
                                        e.preventDefault() // 阻止默认的表单提交行为
                                        handleSubmit()
                                    }
                                }}
                                className="w-full h-8 rounded-full bg-gray-100 border border-gray-100 focus:outline-none px-8"
                                placeholder="输入你的问题…"
                            />
                            <button
                                type="button"
                                className="absolute right-4 bg-green-100 w-8 h-8 rounded-full bg-green-100"
                                onClick={handleSubmit}
                            >
                                ↑
                            </button>
                            <p className="text-xs">
                                系统的回答仅供参考，重要资讯请咨询专业医生
                            </p>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    )
}
export default Chat

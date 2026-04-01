import React, {
    useState,
    useEffect,
    useRef
} from "react";

const TypingEffectMessage = ({
    text
}) => {
    const [displayedText, setDisplayedText] = useState("");
    const indexRef = useRef(0);
    useEffect(() => {
        setDisplayedText(""); // 重置显示内容


        if (!text) return; // 如果 text 是 undefined/null，则不执行打字效果
        const interval = setInterval(() => {
            const index = indexRef.current;
            if (index < text.length) {
                setDisplayedText((prev) => prev + text.charAt(index));
                console.log("now index", index, text.charAt(index))
                indexRef.current++;

            } else {
                clearInterval(interval);
            }
        }, 30); // 调整时间控制打字速度

        return () => clearInterval(interval); // 清理定时器
    }, [text]);

    return <p className="break-words whitespace-normal">{displayedText}</p>;
};

export default TypingEffectMessage;
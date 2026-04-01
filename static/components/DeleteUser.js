import { useState, useEffect } from "react"

const DeleteUser = () => {
    const [credentials, setCredentials] = useState({})
    const [username, setUsername] = useState("")

    useEffect(() => {
        const currentUsername = localStorage.getItem("now_username")
        setUsername(currentUsername)

        const storedCredentials = localStorage.getItem("user_credentials")
        if (storedCredentials) {
            setCredentials(JSON.parse(storedCredentials))
        }
    }, [])

    const handleDeleteUser = () => {
        const { [username]: _, ...remainingCredentials } = credentials

        localStorage.setItem(
            "user_credentials",
            JSON.stringify(remainingCredentials)
        )
        localStorage.removeItem("now_username")
        setCredentials(remainingCredentials)
        alert("用户已删除成功！")
    }

    return (
        <div className="flex items-center justify-center">
            <p>你确定要删除用户 {username} 吗？</p>
            <button
                onClick={handleDeleteUser}
                className="h-8 bg-gray-200 border rounded-lg p-2 flex items-center justify-center"
            >
                删除用户
            </button>
        </div>
    )
}

export default DeleteUser

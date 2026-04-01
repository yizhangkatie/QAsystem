import { useState, useEffect } from "react"

const ChangePassword = ({ closeModal }) => {
    const [credentials, setCredentials] = useState({})
    const [username, setUsername] = useState("")
    const [oldPassword, setOldPassword] = useState("")
    const [newPassword, setNewPassword] = useState("")

    useEffect(() => {
        const currentUsername = localStorage.getItem("now_username")
        setUsername(currentUsername)

        const storedCredentials = localStorage.getItem("user_credentials")
        if (storedCredentials) {
            setCredentials(JSON.parse(storedCredentials))
        }
    }, [])

    const handleChangePassword = () => {
        const user = credentials[username]
        if (user && user.password === oldPassword && newPassword) {
            const updatedCredentials = {
                ...credentials,
                [username]: { ...user, password: newPassword },
            }
            localStorage.setItem(
                "user_credentials",
                JSON.stringify(updatedCredentials)
            )
            setCredentials(updatedCredentials)
            alert("密码修改成功！")
        }
        if (!newPassword) {
            alert("请输入新密码。")
        } else {
            alert("旧密码不正确，请重试。")
        }
    }

    return (
        <div className="fixed inset-0 flex items-center justify-center z-50">
            {/* <div
          className="fixed inset-0 bg-black opacity-50"
          onClick={closeModal} 
        > 
        </div> */}

            <div className="relative flex flex-col space-y-12 items-center bg-white border border-gray-100 shadow-lg w-1/3 h-auto p-8 z-50 rounded-lg">
                <button
                    className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
                    onClick={closeModal}
                    style={{ fontSize: "24px" }}
                >
                    &times;
                </button>

                <h1>修改密码</h1>
                <input
                    type="password"
                    placeholder="旧密码"
                    value={oldPassword}
                    onChange={(e) => setOldPassword(e.target.value)}
                    className="w-2/3 h-8 bg-gray-200 border rounded-lg p-2"
                />
                <input
                    type="password"
                    placeholder="新密码"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="w-2/3 h-8 bg-gray-200 border rounded-lg p-2"
                />
                <button
                    className="w-16 bg-mypurple100 border rounded-lg p-2"
                    onClick={handleChangePassword}
                >
                    提交
                </button>
            </div>
        </div>
    )
}

export default ChangePassword

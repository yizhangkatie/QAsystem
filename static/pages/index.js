import React, { useState, useEffect } from "react"
import { useRouter } from "next/router"
import FirstPage from "../components/firstpage"

const Home = () => {
    // localStorage.clear(); //清除所有用户数据
    const [isLogin, setIsLogin] = useState(true)
    const [username, setUsername] = useState("")
    const [password, setPassword] = useState("")
    const [credentials, setCredentials] = useState({})
    const router = useRouter()

    useEffect(() => {
        const storedCredentials =
            JSON.parse(localStorage.getItem("user_credentials")) || {}
        setCredentials(storedCredentials)
    }, [])

    const handleSubmit = (e) => {
        e.preventDefault()
    }

    // const userCredentials = localStorage.getItem('user_credentials');
    // if (userCredentials) {
    //     console.log(JSON.parse(userCredentials));
    // } else {
    //     console.log('没有存储的用户凭据');
    // }

    const handleRegister = () => {
        if (credentials[username]) {
            alert("用户名已存在，请登录或选择其他用户名")
            return
        }

        const newCredentials = {
            ...credentials,
            [username]: { password, is_admin: false },
        }

        localStorage.setItem("user_credentials", JSON.stringify(newCredentials))
        localStorage.setItem("now_username", username)
        setCredentials(newCredentials)
        alert("注册成功！")
        setIsLogin(true)
    }

    const handleLogin = () => {
        const user = credentials[username]

        if (user && user.password === password) {
            localStorage.setItem("now_username", username)
            alert("登录成功！")
            router.push("/post/chat")
        } else if (!user) {
            alert("用户不存在，请先注册。")
            setIsLogin(false)
        } else {
            alert("用户名或密码错误。")
        }
    }

    return (
        <main className="w-full h-screen flex items-center justify-center homepage-bg">
            <div className="w-1/2 h-3/5 bg-white opacity-90 rounded-lg flex items-center justify-center">
                <FirstPage />
                <form
                    onSubmit={handleSubmit}
                    className="w-1/2 h-full p-6 flex flex-col items-center"
                >
                    <p className="text-2xl pb-8 pt-12">
                        {isLogin ? "登录" : "注册"}
                    </p>
                    <div className="w-4/5 flex flex-col justify-start pb-6 pl-3">
                        <label htmlFor="username" className="text-base pb-2">
                            {isLogin ? "用户名" : "设置用户名"}：
                        </label>
                        <input
                            type="text"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            className="h-8 bg-gray-200 border rounded-lg p-2"
                            placeholder="输入2~12个字符"
                        />

                        <label
                            htmlFor="password"
                            className="text-base pb-2 pt-4"
                        >
                            {isLogin ? "密码" : "设置密码"}：
                        </label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className="h-8 bg-gray-200 border rounded-lg p-2"
                            autoComplete="off"
                            placeholder="输入6~18个字符"
                        />
                    </div>
                    <button
                        type="submit"
                        onClick={isLogin ? handleLogin : handleRegister}
                        className="w-24 h-8 rounded-lg bg-blue-300"
                    >
                        {isLogin ? "登录" : "注册"}
                    </button>
                    <button
                        type="button"
                        onClick={() => setIsLogin(!isLogin)}
                        className="text-xs pt-4"
                    >
                        {isLogin ? "没有账号，去注册" : "已有账号，去登录"}
                    </button>
                </form>
            </div>
        </main>
    )
}

export default Home

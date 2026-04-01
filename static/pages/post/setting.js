import React from "react"
import LeftBotton from "../../components/left_button"
import TopCol from "../../components/top_col"
import { useEffect, useState } from "react"
import ChangePassword from "../../components/ChangePassword"
import DeleteUser from "../../components/DeleteUser"

export default function Setting() {
    const [nowUsername, setNowUsername] = useState("")
    const [isModalOpen, setModalOpen] = useState(false)

    useEffect(() => {
        const storedNowUsername = localStorage.getItem("now_username")
        if (storedNowUsername) {
            setNowUsername(storedNowUsername)
        }
    }, [])
    return (
        <div className="relative flex h-screen w-full justify-center">
            <LeftBotton />
            <TopCol col_name={"账号设置"} />
            <div className="absolute top-12 flex flex-col pt-12 pl-16 w-full items-center">
                <div className="relative flex flex-col w-1/4 h-2/3 items-center justify-center space-y-24">
                    <div className="flex text-base w-full space-x-6">
                        <p>账号：</p>
                        <div className="absolute right-0 pl-8 flex flex-col space-y-4">
                            <div className="flex justify-center items-center w-48 h-8 bg-gray-200 border border-gray rounded-lg p-2">
                                {nowUsername}
                            </div>

                            <button
                                onClick={() => setModalOpen(!isModalOpen)}
                                className="border border-gray-200 rounded-lg w-24 h-8 px-2"
                            >
                                修改密码
                            </button>
                            {isModalOpen && (
                                <ChangePassword
                                    closeModal={() => setModalOpen(false)}
                                />
                            )}
                        </div>
                    </div>
                    <div className="flex text-base w-full space-x-6">
                        <p>当前模式：</p>
                        <div className="absolute right-0 flex flex-col space-y-4 ">
                            <div className="flex justify-center items-center w-48 h-8 bg-gray-200 border border-gray rounded-lg p-2">
                                普通模式
                            </div>
                            <button className="border border-gray-200 rounded-lg w-32 h-8 px-2">
                                切换关怀模式
                            </button>
                        </div>
                    </div>
                    <div className="flex text-base w-full space-x-6">
                        <p>当前语言：</p>
                        <div className="absolute right-0  flex flex-col space-y-4 ">
                            <div className="flex justify-center items-center w-48 h-8 bg-gray-200 border border-gray rounded-lg p-2">
                                简体中文
                            </div>
                            <button className="border border-gray-200 rounded-lg w-24 h-8 px-2">
                                切换语言
                            </button>
                        </div>
                    </div>
                    <button
                        onClick={() => (window.location.href = "/")}
                        className="bg-mypurple100 rounded-lg w-32 h-12"
                    >
                        退出登录
                    </button>
                </div>
                {/* <button 
                className="border border-gray-200 rounded-lg w-24 h-8">
                    删除用户
                </button> */}

                {/* {activeComponent === 'changePassword' && <ChangePassword />} */}
                {/* {activeComponent === 'deleteUser' && <DeleteUser />} */}
            </div>
        </div>
    )
}

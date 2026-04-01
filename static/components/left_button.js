import React from "react"

export default function LeftBotton() {
    return (
        <div className="flex flex-col w-16 h-full bg-white border border-gray-100 items-center space-y-6 z-10">
            <button
                onClick={() => (window.location.href = "/post/chat")}
                className="w-6 h-6 bg-white rounded-full text-black mt-12"
            >
                <img src="/images/home.png" alt="settings" className="object-contain"></img>
            </button>
            <button
                onClick={() => (window.location.href = "/post/setting")}
                className="w-6 h-6 bg-white rounded-full text-black"
            >
                <img src="/images/settings.png" alt="settings" className="object-contain"></img>
            </button>
            <button
                onClick={() => (window.location.href = "/post/help")}
                className="w-6 h-6 bg-white rounded-full text-black"
            >
                <img src="/images/help.png" alt="settings" className="object-cover"></img>
            </button>
        </div>
    )
}

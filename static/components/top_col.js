import React from "react"

export default function TopCol({ col_name }) {
    return (
        <div className="relative flex flex-1 w-full h-12 z-10">
            <div className="absolute top-0 w-full h-12 bg-white border border-gray-200 flex items-center justify-center">
                <p className="absolute top-4 text-xs">{col_name}</p>
            </div>
        </div>
    )
}

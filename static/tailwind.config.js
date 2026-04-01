/** @type {import('tailwindcss').Config} */

module.exports = {
    content: [
        "./pages/**/*.{js,jsx,ts,tsx}",
        "./components/**/*.{js,jsx,ts,tsx}",
    ],
    important: true,
    theme: {
        extend: {
            zIndex: {
                max: "1000000000",
            },
            colors: {
                primary: "#29525a",
                cornflowerblue: "#6495ed",
                mypurple200: "#9999ff",
                mypurple100: "#ccccff",
            },
            width: {
                content: "calc(100% - 496px)",
            },
            height: {
                content: "calc(100vh - 72px)",
                mcontent: "calc(100vh - 114px)",
            },
            backgroundImage: {
                "radial-gradient":
                    "radial-gradient(circle, var(--tw-gradient-stops))",
            },
            keyframes: {},
            animation: {},
        },
    },
    plugins: [],
}

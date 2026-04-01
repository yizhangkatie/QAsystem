import "../styles/global.css"

export default function App({ Component, pageProps }) {
    console.info(Component)
    return (
        <>
            <Component {...pageProps} />
        </>
    )
}

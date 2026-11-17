"use client";
export default function ErrorPage({ reset }: { reset: () => void }) { return <main id="main" className="standalone"><h1>The match view could not load</h1><p>Your local files have not been uploaded. Reload the view to try again.</p><button className="primary" onClick={reset}>Reload match view</button></main>; }

import type { Metadata } from "next";
import "@fontsource/inter/400.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@possession/design-tokens/tokens.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Possession | Barcelona vs Athletic Club",
  description: "Review five minutes of ball possession, compare reference labels with model results, and open match video privately on your device."
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><a className="skip-link" href="#main">Skip to content</a>{children}</body></html>;
}

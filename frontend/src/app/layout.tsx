import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "资源推动 Agent",
  description: "企业资源调查与内部项目检索"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}


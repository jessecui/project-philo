"use client";

import Cookies from "js-cookie";
import { Loader2, Search } from "lucide-react";
import { useEffect, useState } from "react";

import { AuthModal } from "@/components/auth-modal";
import { cn } from "@/lib/utils";

interface Source {
  filename: string;
  paragraph_idx: number;
  text: string;
  score: number | null;
}

export default function HomePage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    // Check if user is already authenticated
    const authCookie = Cookies.get("creator_auth");
    if (authCookie === "true") {
      setIsAuthenticated(true);
    }
  }, []);

  const handleAuthenticated = () => {
    setIsAuthenticated(true);
  };

  const formatAnswer = (text: string) => {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    // Match **bold** and *italic* patterns
    const regex = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      const matchedText = match[0];
      if (matchedText.startsWith("**") && matchedText.endsWith("**")) {
        // Bold text
        parts.push(
          <strong key={match.index}>{matchedText.slice(2, -2)}</strong>,
        );
      } else if (matchedText.startsWith("*") && matchedText.endsWith("*")) {
        // Italic text
        parts.push(<em key={match.index}>{matchedText.slice(1, -1)}</em>);
      }

      lastIndex = regex.lastIndex;
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts;
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setAnswer("");
    setSources([]);
    setError("");

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/search-and-generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          top_k_context: 5,
          use_reranking: true,
          top_k_faiss: 50,
          context_window: 2,
          temperature: 0.7,
          max_output_tokens: 8192,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body");
      }

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));

            if (data.type === "sources") {
              setSources(data.data);
            } else if (data.type === "token") {
              setAnswer((prev) => prev + data.data);
            } else if (data.type === "error") {
              setError(data.data.message);
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (exampleQuery: string) => {
    setQuery(exampleQuery);
  };

  return (
    <>
      {!isAuthenticated && <AuthModal onAuthenticated={handleAuthenticated} />}
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 p-4">
        <div className="max-w-4xl mx-auto space-y-8 py-12">
          {/* Header */}
          <div className="text-center space-y-4">
            <h1 className="text-5xl font-bold text-white">Project Philo</h1>
            <p className="text-slate-400 text-lg">
              Your AI philosophy assistant, powered by semantic search and
              intelligent reranking.
            </p>
          </div>

          {/* Search Box */}
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-lg p-8 space-y-6">
            <form onSubmit={handleSearch} className="flex gap-3">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question regarding philosophy..."
                className={cn(
                  "flex-1 px-4 py-3 bg-slate-900 border border-slate-700",
                  "rounded-lg text-white placeholder-slate-500",
                  "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                  "transition-all",
                )}
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !query.trim()}
                className={cn(
                  "px-6 py-3 rounded-lg flex items-center gap-2 font-medium transition-colors",
                  "bg-blue-600 hover:bg-blue-700 text-white",
                  "disabled:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50",
                )}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Searching
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    Search
                  </>
                )}
              </button>
            </form>

            {/* Quick Stats */}
            <div className="flex gap-6 text-sm text-slate-400 justify-center pt-4 border-t border-slate-700">
              <div>
                <span className="font-semibold text-white">FAISS</span> Vector
                Search
              </div>
              <div>•</div>
              <div>
                <span className="font-semibold text-white">Cross-Encoder</span>{" "}
                Reranking
              </div>
              <div>•</div>
              <div>
                <span className="font-semibold text-white">Gemini 2.5</span>{" "}
                Generation
              </div>
            </div>
          </div>

          {/* Example Questions */}
          {!answer && !isLoading && (
            <div className="space-y-3">
              <p className="text-slate-500 text-sm text-center">Try asking:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {[
                  "What is self-reliance?",
                  "How should I live according to the Tao?",
                  "How do we accept what's not in our control?",
                ].map((question) => (
                  <button
                    key={question}
                    onClick={() => handleExampleClick(question)}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-full text-sm text-slate-300 transition-colors"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
              <p className="text-red-400">{error}</p>
            </div>
          )}

          {/* Answer */}
          {answer && (
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-lg p-8 space-y-6">
              <div>
                <h2 className="text-xl font-semibold text-white mb-4">
                  Answer
                </h2>
                <div className="text-slate-300 leading-relaxed whitespace-pre-line">
                  {formatAnswer(answer)}
                </div>
              </div>

              {/* Sources */}
              {sources.length > 0 && (
                <div className="border-t border-slate-700 pt-6 space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    Sources ({sources.length})
                  </h3>
                  <div className="space-y-3">
                    {sources.map((source, idx) => (
                      <div
                        key={idx}
                        className="bg-slate-900/50 border border-slate-700 rounded-lg p-4 space-y-2"
                      >
                        <div className="flex justify-between items-start">
                          <div className="font-medium text-blue-400">
                            {source.filename
                              .replace(/\.[^/.]+$/, "")
                              .replace(/_/g, " ")}
                          </div>
                          {source.score !== null && (
                            <div className="text-xs text-slate-500">
                              Score: {source.score.toFixed(3)}
                            </div>
                          )}
                        </div>
                        <div className="text-sm text-slate-400">
                          Paragraph {source.paragraph_idx}
                        </div>
                        <div className="text-slate-300 text-sm leading-relaxed">
                          {source.text}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

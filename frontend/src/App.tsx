import Cookies from "js-cookie";
import { ChevronDown, Loader2, Search } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { AuthModal } from "@/components/auth-modal";
import { cn } from "@/lib/utils";

interface Source {
  filename: string;
  author: string | null;
  paragraph_idx: number;
  text: string;
  score: number | null;
}

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<"searching" | "thinking" | "generating">("searching");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState("");
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const [highlightedSource, setHighlightedSource] = useState<number | null>(null);

  useEffect(() => {
    const authCookie = Cookies.get("creator_auth");
    if (authCookie === "true") {
      setIsAuthenticated(true);
    }
  }, []);

  // Clear highlight on click anywhere
  useEffect(() => {
    if (highlightedSource === null) return;

    const clearHighlight = () => setHighlightedSource(null);

    // Use capture phase to catch click before it bubbles
    window.addEventListener("click", clearHighlight, { capture: true });

    return () => window.removeEventListener("click", clearHighlight, { capture: true });
  }, [highlightedSource]);

  const scrollToSource = useCallback((sourceNum: number) => {
    // Validate source exists
    if (sourceNum < 1 || sourceNum > sources.length) return;

    // Expand sources if collapsed
    setSourcesExpanded(true);

    // Small delay to allow expansion
    setTimeout(() => {
      const element = document.getElementById(`source-${sourceNum}`);
      if (element) {
        setHighlightedSource(sourceNum);
        element.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 100);
  }, [sources.length]);

  const handleAuthenticated = () => {
    setIsAuthenticated(true);
  };

  const formatAnswer = (text: string) => {
    // Split into paragraphs (double newlines)
    const paragraphs = text.split(/\n\n+/);

    return paragraphs.map((paragraph, pIdx) => {
      // Check if paragraph is a list (starts with - or * or number.)
      const lines = paragraph.split(/\n/);
      const isList = lines.every(
        (line) => /^[-*•]\s/.test(line.trim()) || /^\d+[.)]\s/.test(line.trim()) || line.trim() === ""
      );

      if (isList && lines.some((l) => l.trim())) {
        const listItems = lines.filter((line) => line.trim());
        return (
          <ul key={pIdx} className="list-disc list-inside space-y-2 my-4 marker:text-slate-400">
            {listItems.map((item, idx) => (
              <li key={idx} className="text-slate-300 leading-relaxed">
                {formatInlineText(item.replace(/^[-*•]\s*/, "").replace(/^\d+[.)]\s*/, ""))}
              </li>
            ))}
          </ul>
        );
      }

      // Regular paragraph
      return (
        <p key={pIdx} className={pIdx > 0 ? "mt-4" : ""}>
          {formatInlineText(paragraph)}
        </p>
      );
    });
  };

  const formatInlineText = useCallback((text: string) => {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    // Match **bold**, *italic*, and [n] citations
    const regex = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\d+\])/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      const matchedText = match[0];
      if (matchedText.startsWith("**") && matchedText.endsWith("**")) {
        parts.push(
          <strong key={match.index} className="font-semibold text-white">
            {matchedText.slice(2, -2)}
          </strong>,
        );
      } else if (matchedText.startsWith("*") && matchedText.endsWith("*")) {
        parts.push(<em key={match.index}>{matchedText.slice(1, -1)}</em>);
      } else if (/^\[\d+\]$/.test(matchedText)) {
        // Citation marker - check if valid source exists
        const num = parseInt(matchedText.slice(1, -1));
        if (num >= 1 && num <= sources.length) {
          parts.push(
            <button
              key={match.index}
              onClick={() => scrollToSource(num)}
              className="text-blue-400 hover:text-blue-300 hover:underline cursor-pointer font-medium"
            >
              {matchedText}
            </button>
          );
        } else {
          // Invalid citation - render as plain text
          parts.push(matchedText);
        }
      }

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts;
  }, [sources.length, scrollToSource]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setLoadingPhase("searching");
    setAnswer("");
    setSources([]);
    setError("");

    try {
      const apiUrl = import.meta.env.VITE_API_URL || "/api";
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
              setLoadingPhase("thinking");
            } else if (data.type === "token") {
              setLoadingPhase("generating");
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

  const handleReset = () => {
    setQuery("");
    setAnswer("");
    setSources([]);
    setError("");
    setSourcesExpanded(false);
    setHighlightedSource(null);
  };

  return (
    <>
      {!isAuthenticated && <AuthModal onAuthenticated={handleAuthenticated} />}
      <div className="min-h-screen bg-linear-to-b from-[#010816] via-slate-950 to-slate-900 p-4">
        <div className="max-w-4xl mx-auto space-y-8 py-12">
          {/* Header */}
          <div className="text-center space-y-4">
            <h1
              onClick={handleReset}
              className="text-5xl font-bold text-white cursor-pointer hover:text-slate-200 transition-colors"
            >
              Project Philo
            </h1>
            <p className="text-slate-400/80 text-lg">
              Search classic philosophy, powered by AI.
            </p>
          </div>

          {/* Search Box */}
          <div className="bg-slate-900/80 backdrop-blur border border-slate-700 rounded-lg p-8 space-y-6">
            <form onSubmit={handleSearch} className="flex gap-3">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question regarding philosophy..."
                className={cn(
                  "flex-1 px-4 py-3 bg-slate-950 border border-slate-700",
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
                  "px-6 py-3 rounded-lg flex items-center gap-2 font-medium transition-colors cursor-pointer",
                  "bg-blue-600 hover:bg-blue-700 text-white",
                  "disabled:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50",
                )}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    {loadingPhase === "searching" && "Finding sources..."}
                    {loadingPhase === "thinking" && "Thinking..."}
                    {loadingPhase === "generating" && "Generating..."}
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    Search
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Example Questions */}
          {!answer && !isLoading && (
            <div className="space-y-3">
              <p className="text-slate-400 text-sm text-center">Try asking:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {[
                  "What is self-reliance?",
                  "How should I live according to the Tao?",
                  "How do we accept what's not in our control?",
                ].map((question) => (
                  <button
                    key={question}
                    onClick={() => handleExampleClick(question)}
                    className="px-4 py-2 bg-slate-900/80 hover:bg-slate-800/80 border border-slate-700 hover:border-slate-600 rounded-full text-sm text-slate-300 transition-colors cursor-pointer"
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
            <div className="bg-slate-900/80 backdrop-blur border border-slate-700 rounded-lg p-8 space-y-6">
              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-6">
                <div className="text-slate-300 leading-7">
                  {formatAnswer(answer)}
                </div>
              </div>

              {/* Sources */}
              {sources.length > 0 && (
                <div className="border-t border-slate-700 pt-6 space-y-4">
                  <button
                    onClick={() => setSourcesExpanded(!sourcesExpanded)}
                    className="flex items-center gap-2 text-lg font-semibold text-white hover:text-slate-300 transition-colors cursor-pointer"
                  >
                    <ChevronDown
                      className={cn(
                        "w-5 h-5 transition-transform",
                        sourcesExpanded ? "rotate-0" : "-rotate-90"
                      )}
                    />
                    Sources ({sources.length})
                  </button>
                  {sourcesExpanded && (
                    <div className="space-y-3">
                      {sources.map((source, idx) => (
                        <div
                          key={idx}
                          id={`source-${idx + 1}`}
                          className={cn(
                            "bg-slate-950/80 border rounded-lg p-6 space-y-2 transition-all duration-300",
                            highlightedSource === idx + 1
                              ? "border-blue-400 ring-2 ring-blue-400/50 shadow-lg shadow-blue-500/20"
                              : "border-slate-800"
                          )}
                        >
                          <div className="flex justify-between items-start">
                            <div className="flex items-center gap-2">
                              <span className="text-blue-400 font-semibold">[{idx + 1}]</span>
                              <span className="font-medium text-blue-400">
                                {source.filename
                                  .replace(/\.txt$/, "")
                                  .split("__")[0]
                                  .replace(/_/g, " ")}
                              </span>
                              {source.author && (
                                <span className="text-slate-400">by {source.author}</span>
                              )}
                            </div>
                            {source.score !== null && (
                              <div className="text-xs text-slate-400">
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
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

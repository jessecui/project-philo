"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Loader2 } from "lucide-react";
import Cookies from "js-cookie";

interface AuthModalProps {
  onAuthenticated: () => void;
}

export function AuthModal({ onAuthenticated }: AuthModalProps) {
  const [answer, setAnswer] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/validate-creator`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ answer }),
      });

      if (!response.ok) {
        throw new Error("Failed to validate answer");
      }

      const data = await response.json();

      if (data.valid) {
        // Set cookie with 1 day expiration
        Cookies.set("creator_auth", "true", { expires: 1 });
        onAuthenticated();
      } else {
        setError("Incorrect answer. Please try again.");
        setAnswer("");
      }
    } catch (err) {
      setError("Network error. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={true} modal>
      <DialogContent
        className="sm:max-w-md bg-slate-900 text-white"
        onInteractOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="text-xl text-white">Welcome to Project Philo!</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label htmlFor="answer" className="text-sm font-medium mb-1 block">
              Who created this app?
            </label>
            <input
              id="answer"
              type="text"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              disabled={isLoading}
              className="w-full px-3 py-2 border border-slate-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 bg-slate-900 text-white placeholder-slate-500 mt-1"
              placeholder="Enter your answer"
              autoFocus
            />
          </div>
          {error && (
            <p className="text-sm text-red-500">{error}</p>
          )}
          <button
            type="submit"
            disabled={isLoading || !answer.trim()}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Verifying...
              </>
            ) : (
              "Submit"
            )}
          </button>
        </form>
      </DialogContent>
    </Dialog>
  );
}

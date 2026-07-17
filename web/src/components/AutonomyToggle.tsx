import { useEffect, useRef, useState } from "react";
import * as client from "../api/client";

type Autonomy = "guarded" | "autonomous";

interface AutonomyToggleProps {
  threadId: string | null;
}

const DESCRIPTIONS: Record<Autonomy, string> = {
  guarded: "Asks for approval before training, promoting, deleting, or monitor actions.",
  autonomous: "Acts on training, promoting, deleting, and monitor actions without waiting for approval.",
};

const CONFIRM_MS = 2200;

export function AutonomyToggle({ threadId }: AutonomyToggleProps) {
  const [value, setValue] = useState<Autonomy>("guarded");
  const [hovered, setHovered] = useState<Autonomy | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const confirmTimer = useRef<number | null>(null);

  useEffect(() => {
    if (!threadId) {
      setValue("guarded");
      setError(null);
      return;
    }
    let ignore = false;
    client
      .getSnapshot(threadId)
      .then((snapshot) => {
        if (ignore) return;
        if (snapshot.autonomy === "guarded" || snapshot.autonomy === "autonomous") {
          setValue(snapshot.autonomy);
        }
        setError(null);
      })
      .catch((err) => {
        if (!ignore) setError(String(err));
      });
    return () => {
      ignore = true;
    };
  }, [threadId]);

  useEffect(() => () => {
    if (confirmTimer.current) window.clearTimeout(confirmTimer.current);
  }, []);

  async function setAutonomy(next: Autonomy) {
    if (!threadId || loading) return;
    if (next === value) return;
    setLoading(true);
    try {
      const result = await client.setAutonomy(threadId, next);
      if (result.autonomy === "guarded" || result.autonomy === "autonomous") {
        setValue(result.autonomy);
        setConfirmed(true);
        if (confirmTimer.current) window.clearTimeout(confirmTimer.current);
        confirmTimer.current = window.setTimeout(() => setConfirmed(false), CONFIRM_MS);
      }
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  const displayed = hovered ?? value;

  return (
    <div className="autonomy-bar">
      <div className="autonomy-bar-label">
        <strong>Agent mode</strong>
        {error ? (
          <span className="control-error">{error}</span>
        ) : (
          <span className={confirmed && !hovered ? "confirmed" : undefined}>{DESCRIPTIONS[displayed]}</span>
        )}
      </div>
      <div className="segmented" aria-label="Agent autonomy">
        {(["guarded", "autonomous"] as const).map((option) => (
          <button
            type="button"
            key={option}
            onClick={() => setAutonomy(option)}
            onMouseEnter={() => setHovered(option)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(option)}
            onBlur={() => setHovered(null)}
            disabled={!threadId || loading}
            aria-pressed={value === option}
          >
            {option[0].toUpperCase() + option.slice(1)}
          </button>
        ))}
      </div>
    </div>
  );
}

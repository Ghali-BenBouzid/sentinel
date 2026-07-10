import { useEffect, useState } from "react";
import * as client from "../api/client";

type Autonomy = "guarded" | "autonomous";

interface AutonomyToggleProps {
  threadId: string | null;
}

export function AutonomyToggle({ threadId }: AutonomyToggleProps) {
  const [value, setValue] = useState<Autonomy>("guarded");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  async function toggle() {
    if (!threadId || loading) return;
    const next: Autonomy = value === "guarded" ? "autonomous" : "guarded";
    setLoading(true);
    try {
      const result = await client.setAutonomy(threadId, next);
      if (result.autonomy === "guarded" || result.autonomy === "autonomous") {
        setValue(result.autonomy);
      }
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="autonomy-control">
      <button type="button" onClick={toggle} disabled={!threadId || loading}>
        {value === "guarded" ? "Guarded" : "Autonomous"}
      </button>
      {error ? <span className="control-error">{error}</span> : null}
    </div>
  );
}

import type { Pending } from "../state/useSession";

interface ConfirmCardProps {
  pending: Pending;
  onAnswer: (value: "yes" | "no") => void;
  busy: boolean;
}

export function ConfirmCard({ pending, onAnswer, busy }: ConfirmCardProps) {
  return (
    <div className="confirm">
      <div>
        <strong>{pending.tool}</strong>
        <p>{pending.detail}</p>
      </div>
      <div className="confirm-actions">
        <button type="button" onClick={() => onAnswer("yes")} disabled={busy}>
          Yes
        </button>
        <button type="button" onClick={() => onAnswer("no")} disabled={busy}>
          No
        </button>
      </div>
    </div>
  );
}

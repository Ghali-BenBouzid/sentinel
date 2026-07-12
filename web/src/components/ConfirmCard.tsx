import type { Pending } from "../state/useSession";

interface ConfirmCardProps {
  pending: Pending;
  onChoose: (value: "yes" | "no") => void;
  disabled: boolean;
}

export function ConfirmCard({ pending, onChoose, disabled }: ConfirmCardProps) {
  return (
    <div className="confirm-option">
      <div>
        <strong>{pending.tool}</strong>
        <p>{pending.detail}</p>
      </div>
      <div className="confirm-actions">
        <button
          type="button"
          onClick={() => onChoose("yes")}
          disabled={disabled}
          aria-pressed={pending.decision === "yes"}
          className={pending.decision === "yes" ? "chosen" : undefined}
        >
          Yes
        </button>
        <button
          type="button"
          onClick={() => onChoose("no")}
          disabled={disabled}
          aria-pressed={pending.decision === "no"}
          className={pending.decision === "no" ? "chosen" : undefined}
        >
          No
        </button>
      </div>
    </div>
  );
}

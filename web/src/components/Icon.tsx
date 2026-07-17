interface IconProps {
  name:
    | "agent"
    | "alert"
    | "chart"
    | "chevron"
    | "database"
    | "experiment"
    | "integrations"
    | "model"
    | "monitor"
    | "panel"
    | "plus"
    | "send"
    | "settings"
    | "spark"
    | "status";
  size?: number;
}

const paths: Record<IconProps["name"], React.ReactNode> = {
  agent: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <circle cx="8" cy="8" r="1.6" fill="currentColor" />
    </>
  ),
  alert: <path d="M8 2.2c-1.6 0-2.8 1.5-2.8 3.7 0 1.5 0 3.7-2 4.5h9.6c-2-.8-2-3-2-4.5 0-2.2-1.2-3.7-2.8-3.7ZM6.4 12.2c.2 1 1 1.6 1.6 1.6s1.4-.6 1.6-1.6" />,
  chart: <path d="M2 12.5V9h3v3.5M6.5 12.5V5.5h3v7M11 12.5V2h3v10.5" />,
  chevron: <path d="m10 3-5 5 5 5" />,
  database: (
    <>
      <ellipse cx="8" cy="4" rx="5.5" ry="2.1" />
      <path d="M2.5 4v8c0 1.2 2.5 2.1 5.5 2.1s5.5-.9 5.5-2.1V4M2.5 8c0 1.2 2.5 2.1 5.5 2.1s5.5-.9 5.5-2.1" />
    </>
  ),
  experiment: <path d="M6.3 2.2h3.4v4.2l3.3 6c.5 1-.2 2.2-1.4 2.2H4.4c-1.2 0-1.9-1.2-1.4-2.2l3.3-6ZM5.6 9.5h4.8" />,
  integrations: <path d="M6 1.5h4v4H6zM6 10.5h4v4H6zM8 5.5v5" />,
  model: <path d="m8 1.6 5.6 2.8v7.2L8 14.4l-5.6-2.8V4.4ZM2.4 4.4 8 7.2l5.6-2.8M8 7.2v7.2" />,
  monitor: <path d="M1 8.5h3.5L6 4l2.5 8.5 1.5-4h5" />,
  panel: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="1.5" />
      <path d="M9.5 3v10" />
    </>
  ),
  plus: <path d="M8 2v12M2 8h12" />,
  send: <path d="M2 8h12m-4-4 4 4-4 4" />,
  settings: (
    <>
      <path d="M2 4.5h12" /><circle cx="6" cy="4.5" r="1.3" fill="currentColor" stroke="none" />
      <path d="M2 8h12" /><circle cx="10.5" cy="8" r="1.3" fill="currentColor" stroke="none" />
      <path d="M2 11.5h12" /><circle cx="4.5" cy="11.5" r="1.3" fill="currentColor" stroke="none" />
    </>
  ),
  spark: <path d="m8 1 1.5 4.5L14 7l-4.5 1.5L8 13l-1.5-4.5L2 7l4.5-1.5Z" />,
  status: <path d="m2 11 3.5-4L8 9.5 11 4l3 3" />,
};

export function Icon({ name, size = 16 }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

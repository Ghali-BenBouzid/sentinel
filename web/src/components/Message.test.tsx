import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Message } from "./Message";

describe("Message", () => {
  it("renders agent messages as GitHub Flavored Markdown", () => {
    const html = renderToStaticMarkup(
      <Message
        bubble={{
          id: "b1",
          kind: "agent",
          text: [
            "## Results",
            "",
            "| Model | RMSE |",
            "| --- | ---: |",
            "| Extra Trees | 12.4 |",
            "",
            "- [x] Evaluated",
            "- ~~Deprecated~~",
          ].join("\n"),
        }}
      />,
    );

    expect(html).toContain("<h2>Results</h2>");
    expect(html).toContain("<table>");
    expect(html).toContain("<th>Model</th>");
    expect(html).toContain("<td>Extra Trees</td>");
    expect(html).toContain('type="checkbox"');
    expect(html).toContain("<del>Deprecated</del>");
  });

  it("keeps user messages as plain text", () => {
    const html = renderToStaticMarkup(
      <Message bubble={{ id: "b2", kind: "user", text: "**literal**" }} />,
    );

    expect(html).toContain("**literal**");
    expect(html).not.toContain("<strong>literal</strong>");
  });
});

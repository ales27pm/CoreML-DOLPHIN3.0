import React from "react";
import renderer from "react-test-renderer";
import { describe, expect, it } from "vitest";

import { PriceTag } from "../../tasks/multi_language_cross_integration/react_snapshot/PriceTag";

describe("PriceTag", () => {
  it("renders formatted currency", () => {
    const tree = renderer
      .create(<PriceTag amount={123.45} currency="EUR" locale="de-DE" />)
      .toJSON();
    expect(tree).toMatchSnapshot();
  });

  it("applies strike-through style when requested", () => {
    const tree = renderer
      .create(<PriceTag amount={99.99} strikeThrough />)
      .toJSON();
    expect(tree).toMatchSnapshot();
  });
});

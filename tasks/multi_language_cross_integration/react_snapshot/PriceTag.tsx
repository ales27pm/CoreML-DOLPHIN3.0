import React from "react";

type PriceTagProps = {
  readonly amount: number;
  readonly currency?: string;
  readonly locale?: string;
  readonly strikeThrough?: boolean;
};

const DEFAULT_LOCALE = "en-US";
const DEFAULT_CURRENCY = "USD";

export const PriceTag: React.FC<PriceTagProps> = ({
  amount,
  currency = DEFAULT_CURRENCY,
  locale = DEFAULT_LOCALE,
  strikeThrough = false,
}) => {
  const formatted = React.useMemo(
    () =>
      new Intl.NumberFormat(locale, {
        currency,
        style: "currency",
        currencyDisplay: "symbol",
      }).format(amount),
    [amount, currency, locale],
  );

  if (strikeThrough) {
    return (
      <span
        aria-label={`Original price ${formatted}`}
        style={{ textDecoration: "line-through" }}
      >
        {formatted}
      </span>
    );
  }

  return <span aria-label={`Current price ${formatted}`}>{formatted}</span>;
};

PriceTag.displayName = "PriceTag";

export default PriceTag;

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "@radix-ui/react-slot";
import { BatteryMedium } from "lucide-react";
import type { PowerSplit } from "@/lib/api";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:pointer-events-none disabled:opacity-40 whitespace-nowrap",
  {
    variants: {
      variant: {
        default: "bg-accent text-bg hover:bg-accent/85",
        outline: "border border-border bg-surface hover:bg-surface-2 text-fg",
        ghost: "hover:bg-surface-2 text-muted hover:text-fg",
        danger: "border border-bad/40 text-bad hover:bg-bad/10",
      },
      size: {
        default: "h-9 px-4",
        sm: "h-8 px-3 text-xs",
        icon: "h-8 w-8",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />;
  },
);
Button.displayName = "Button";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-xl border border-border bg-surface", className)} {...props} />;
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-4 pt-4 pb-2", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("text-sm font-semibold text-fg", className)} {...props} />;
}

export function CardBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-4 pb-4", className)} {...props} />;
}

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium",
  {
    variants: {
      tone: {
        default: "bg-surface-2 text-muted",
        good: "bg-good/12 text-good",
        warn: "bg-warn/12 text-warn",
        bad: "bg-bad/12 text-bad",
        accent: "bg-accent/12 text-accent",
      },
    },
    defaultVariants: { tone: "default" },
  },
);

export function Badge({
  className,
  tone,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}

/** Flags a figure that averages battery and AC measurements together.
 *  Apple Silicon throttles ~12% unplugged, which is wider than most of the
 *  model-to-model gaps on this dashboard — so a mixed figure is not wrong so
 *  much as unreadable, and it should say so rather than look precise. */
export function PowerBadge({ power }: { power: PowerSplit | undefined }) {
  if (!power?.mixed) return null;
  const ac = power.ac.decode_tps?.p50;
  const batt = power.battery.decode_tps?.p50;
  const gap = ac && batt ? Math.abs(ac - batt) / Math.max(ac, batt) : null;
  return (
    <Badge
      tone="warn"
      title={
        `Mixes ${power.ac.n} request(s) on AC with ${power.battery.n} on battery` +
        (ac && batt ? ` — ${ac.toFixed(1)} vs ${batt.toFixed(1)} tok/s` : "")
      }
    >
      <BatteryMedium size={11} />
      mixed power{gap != null ? ` · ${(gap * 100).toFixed(0)}%` : ""}
    </Badge>
  );
}

/** A headline number with its unit and an optional caveat.
 *  Numbers without a confidence interval are styled differently from ones
 *  with — you should be able to see which claims are load-bearing. */
export function Stat({
  label,
  value,
  sub,
  hint,
  color,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  hint?: string;
  color?: string;
}) {
  return (
    <div className="min-w-0" title={hint}>
      <div className="text-[11px] uppercase tracking-wide text-faint">{label}</div>
      <div className="tnum mt-0.5 text-lg font-semibold" style={color ? { color } : undefined}>
        {value}
      </div>
      {sub != null && <div className="tnum text-[11px] text-faint">{sub}</div>}
    </div>
  );
}

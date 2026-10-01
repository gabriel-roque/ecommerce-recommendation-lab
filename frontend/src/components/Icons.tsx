import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement>

const Base = ({ children, ...props }: IconProps) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
    {children}
  </svg>
)

export const DatabaseIcon = (props: IconProps) => <Base {...props}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" /></Base>
export const NetworkIcon = (props: IconProps) => <Base {...props}><circle cx="5" cy="5" r="2" /><circle cx="19" cy="12" r="2" /><circle cx="5" cy="19" r="2" /><path d="m7 6 10 5M7 18l10-5" /></Base>
export const SparkIcon = (props: IconProps) => <Base {...props}><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z" /><path d="m19 17 .7 2.3L22 20l-2.3.7L19 23l-.7-2.3L16 20l2.3-.7L19 17Z" /></Base>
export const ArrowIcon = (props: IconProps) => <Base {...props}><path d="M5 12h14M13 6l6 6-6 6" /></Base>
export const RefreshIcon = (props: IconProps) => <Base {...props}><path d="M20 7h-6V1M4 17h6v6" /><path d="M5.1 9A8 8 0 0 1 18.5 5.5L20 7M4 17l1.5 1.5A8 8 0 0 0 18.9 15" /></Base>
export const SearchIcon = (props: IconProps) => <Base {...props}><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></Base>
export const AlertIcon = (props: IconProps) => <Base {...props}><path d="M12 3 2.5 20h19L12 3Z" /><path d="M12 9v5M12 17.5v.5" /></Base>
export const CheckIcon = (props: IconProps) => <Base {...props}><path d="m5 12 4 4L19 6" /></Base>
export const CpuIcon = (props: IconProps) => <Base {...props}><rect x="7" y="7" width="10" height="10" rx="1" /><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3M10 10h4v4h-4z" /></Base>

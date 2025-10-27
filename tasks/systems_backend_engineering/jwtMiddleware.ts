import type { NextFunction, Request, RequestHandler, Response } from "express";
import jwt, {
  type JwtPayload,
  type VerifyErrors,
  type VerifyOptions,
} from "jsonwebtoken";

export interface JwtLogger {
  info(message: string, meta?: Record<string, unknown>): void;
  warn(message: string, meta?: Record<string, unknown>): void;
  error(message: string, meta?: Record<string, unknown>): void;
}

export interface JwtMiddlewareOptions extends VerifyOptions {
  /**
   * Override the header used for bearer token extraction. Defaults to
   * `authorization` to match Express conventions.
   */
  headerName?: string;
  /**
   * Optional structured logger. `console` is used when omitted.
   */
  logger?: JwtLogger;
  /**
   * Property name injected into the request object containing authentication
   * metadata. Defaults to `auth`.
   */
  requestProperty?: string;
  /**
   * Custom token extractor for advanced scenarios (e.g., query parameters or
   * cookies). When provided it overrides header inspection entirely.
   */
  tokenExtractor?: (req: Request) => string | null;
}

export interface AuthContext<Payload = JwtPayload | string> {
  token: string;
  payload: Payload;
  issuedAt?: number;
  expiresAt?: number;
}

export interface AuthenticatedRequest<
  Params = Record<string, string>,
  ResBody = unknown,
  ReqBody = unknown,
  ReqQuery = Record<string, string>,
  Locals extends Record<string, unknown> = Record<string, unknown>,
> extends Request<Params, ResBody, ReqBody, ReqQuery, Locals> {
  auth?: AuthContext;
}

export class JwtMiddlewareError extends Error {
  readonly code:
    | "missing_authorization_header"
    | "invalid_authorization_scheme"
    | "invalid_token";

  constructor(
    code: JwtMiddlewareError["code"],
    message: string,
    cause?: Error,
  ) {
    super(message);
    this.code = code;
    this.name = "JwtMiddlewareError";
    if (cause) {
      this.cause = cause;
    }
  }
}

function extractBearerToken(
  req: Request,
  headerName: string,
  extractor?: (req: Request) => string | null,
): string | null {
  if (extractor) {
    return extractor(req);
  }

  const header = req.headers[headerName];
  if (!header) {
    return null;
  }
  const value = Array.isArray(header) ? (header[0] ?? "") : header;
  if (typeof value !== "string") {
    return null;
  }
  if (!value.toLowerCase().startsWith("bearer ")) {
    throw new JwtMiddlewareError(
      "invalid_authorization_scheme",
      "Authorization header must use the Bearer scheme",
    );
  }
  const token = value.slice(7).trim();
  return token.length > 0 ? token : null;
}

function normaliseOptions(options: JwtMiddlewareOptions): {
  logger: JwtLogger;
  headerName: string;
  requestProperty: string;
  verify: VerifyOptions;
  tokenExtractor?: (req: Request) => string | null;
} {
  const {
    headerName = "authorization",
    logger = console,
    requestProperty = "auth",
    tokenExtractor,
    ...verify
  } = options;

  const algorithms = Array.isArray(verify.algorithms)
    ? verify.algorithms.filter((alg) => typeof alg === "string" && alg.trim() !== "")
    : [];

  return {
    headerName: headerName.toLowerCase(),
    logger,
    requestProperty,
    verify: {
      ...verify,
      algorithms: algorithms.length > 0 ? algorithms : ["HS256"],
    },
    tokenExtractor,
  };
}

function buildAuthContext(
  token: string,
  payload: string | JwtPayload,
): AuthContext {
  let issuedAt: number | undefined;
  let expiresAt: number | undefined;
  if (typeof payload === "object" && payload !== null) {
    if (typeof payload.iat === "number") {
      issuedAt = payload.iat;
    }
    if (typeof payload.exp === "number") {
      expiresAt = payload.exp;
    }
  }
  return {
    token,
    payload,
    issuedAt,
    expiresAt,
  };
}

export function createJwtMiddleware(
  secret: string,
  options: JwtMiddlewareOptions = {},
): RequestHandler {
  if (typeof secret !== "string" || secret.trim() === "") {
    throw new Error("JWT secret must be a non-empty string");
  }

  const normalised = normaliseOptions(options);

  return (
    req: AuthenticatedRequest,
    res: Response,
    next: NextFunction,
  ): void => {
    let token: string | null = null;

    try {
      token = extractBearerToken(
        req,
        normalised.headerName,
        normalised.tokenExtractor,
      );
    } catch (error) {
      const middlewareError =
        error instanceof JwtMiddlewareError
          ? error
          : new JwtMiddlewareError(
              "invalid_authorization_scheme",
              "Failed to extract bearer token",
              error instanceof Error ? error : undefined,
            );
      normalised.logger.warn(middlewareError.message, {
        code: middlewareError.code,
      });
      res.status(401).json({
        code: middlewareError.code,
        error: middlewareError.message,
      });
      return;
    }

    if (!token) {
      const error = new JwtMiddlewareError(
        "missing_authorization_header",
        "Missing bearer token",
      );
      normalised.logger.warn(error.message, { code: error.code });
      res.status(401).json({ code: error.code, error: error.message });
      return;
    }

    try {
      const payload = jwt.verify(token, secret, normalised.verify);
      const context = buildAuthContext(token, payload);
      (req as AuthenticatedRequest).auth = context;
      if (normalised.requestProperty !== "auth") {
        Object.defineProperty(req, normalised.requestProperty, {
          value: context,
          configurable: true,
          enumerable: false,
          writable: true,
        });
      }
      normalised.logger.info("jwt.verified", {
        subject:
          typeof context.payload === "object"
            ? (context.payload as JwtPayload).sub
            : undefined,
        expiresAt: context.expiresAt,
      });
      next();
    } catch (error) {
      const verifyError = error as VerifyErrors;
      const message =
        verifyError && typeof verifyError.message === "string"
          ? verifyError.message
          : "Invalid token";
      const middlewareError = new JwtMiddlewareError(
        "invalid_token",
        message,
        verifyError instanceof Error ? verifyError : undefined,
      );
      normalised.logger.warn(middlewareError.message, {
        code: middlewareError.code,
      });
      res.status(401).json({
        code: middlewareError.code,
        error: middlewareError.message,
      });
    }
  };
}

export default createJwtMiddleware;

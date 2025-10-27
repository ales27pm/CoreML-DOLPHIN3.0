import express from "express";
import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";
import jwt, { type JwtPayload } from "jsonwebtoken";

import {
  createJwtMiddleware,
  type AuthenticatedRequest,
  type AuthContext,
  type JwtLogger,
} from "../../tasks/systems_backend_engineering/jwtMiddleware";

describe("JWT middleware", () => {
  const secret = "top-secret";
  let logger: JwtLogger;

  beforeEach(() => {
    logger = {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    };
  });

  it("attaches the decoded payload to the request on success", async () => {
    const app = express();
    const middleware = createJwtMiddleware(secret, {
      logger,
      algorithms: ["HS256"],
    });

    app.get("/secure", middleware, (req, res) => {
      const auth = (req as AuthenticatedRequest)
        .auth as AuthContext<JwtPayload>;
      res.json({
        subject: auth.payload.sub,
        issuedAt: auth.issuedAt,
        expiresAt: auth.expiresAt,
      });
    });

    const token = jwt.sign({ sub: "user-123", role: "admin" }, secret, {
      expiresIn: "1h",
    });

    const response = await request(app)
      .get("/secure")
      .set("Authorization", `Bearer ${token}`)
      .expect(200);

    expect(response.body).toMatchObject({
      subject: "user-123",
    });
    expect(logger.info).toHaveBeenCalledWith(
      "jwt.verified",
      expect.objectContaining({ subject: "user-123" }),
    );
  });

  it("returns 401 when the header is missing", async () => {
    const app = express();
    app.get(
      "/secure",
      createJwtMiddleware(secret, { logger, algorithms: ["HS256"] }),
      (_req, res) => {
        res.status(200).json({ ok: true });
      },
    );

    const response = await request(app).get("/secure").expect(401);

    expect(response.body).toEqual({
      code: "missing_authorization_header",
      error: "Missing bearer token",
    });
    expect(logger.warn).toHaveBeenCalledWith("Missing bearer token", {
      code: "missing_authorization_header",
    });
  });

  it("handles expired tokens with structured errors", async () => {
    const app = express();
    const middleware = createJwtMiddleware(secret, {
      logger,
      clockTolerance: 0,
      algorithms: ["HS256"],
    });

    app.get("/secure", middleware, (_req, res) => {
      res.json({ ok: true });
    });

    const expiredToken = jwt.sign({ sub: "user-1" }, secret, {
      expiresIn: "-1s",
    });

    const response = await request(app)
      .get("/secure")
      .set("Authorization", `Bearer ${expiredToken}`)
      .expect(401);

    expect(response.body.code).toBe("invalid_token");
    expect(response.body.error).toContain("expired");
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("expired"),
      {
        code: "invalid_token",
      },
    );
  });

  it("rejects tokens signed with a disallowed algorithm", async () => {
    const app = express();
    const middleware = createJwtMiddleware(secret, {
      logger,
      algorithms: ["HS256"],
    });

    app.get("/secure", middleware, (_req, res) => {
      res.json({ ok: true });
    });

    const token = jwt.sign({ sub: "user-2" }, secret, { algorithm: "HS512" });

    const response = await request(app)
      .get("/secure")
      .set("Authorization", `Bearer ${token}`)
      .expect(401);

    expect(response.body).toEqual({
      code: "invalid_token",
      error: expect.stringContaining("invalid algorithm"),
    });
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("invalid"),
      expect.objectContaining({ code: "invalid_token" }),
    );
  });
});

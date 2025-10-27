package securitylog

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// Event describes a security incident emitted by upstream services.
type Event struct {
	ID        string `json:"id"`
	Timestamp string `json:"timestamp"`
	Severity  string `json:"severity"`
	Message   string `json:"message"`
}

// RetryPolicy controls how the client retries failed transmissions.
type RetryPolicy struct {
	MaxAttempts    int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
}

// Client streams security events to a SIEM endpoint with retry semantics.
type Client struct {
	Endpoint    string
	HTTPClient  *http.Client
	Logger      *log.Logger
	RetryPolicy RetryPolicy
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient == nil {
		c.HTTPClient = &http.Client{Timeout: 10 * time.Second}
	}
	return c.HTTPClient
}

func (c *Client) logger() *log.Logger {
	if c.Logger != nil {
		return c.Logger
	}
	return log.Default()
}

func (c *Client) retryPolicy() RetryPolicy {
	policy := c.RetryPolicy
	if policy.MaxAttempts <= 0 {
		policy.MaxAttempts = 3
	}
	if policy.InitialBackoff <= 0 {
		policy.InitialBackoff = 200 * time.Millisecond
	}
	if policy.MaxBackoff <= 0 {
		policy.MaxBackoff = 5 * time.Second
	}
	if policy.MaxBackoff < policy.InitialBackoff {
		policy.MaxBackoff = policy.InitialBackoff
	}
	return policy
}

func (p RetryPolicy) backoff(attempt int) time.Duration {
	backoff := p.InitialBackoff
	for i := 1; i < attempt; i++ {
		backoff *= 2
		if backoff > p.MaxBackoff {
			return p.MaxBackoff
		}
	}
	if backoff <= 0 {
		return p.InitialBackoff
	}
	if backoff > p.MaxBackoff {
		return p.MaxBackoff
	}
	return backoff
}

// Validate verifies required fields are present.
func (e Event) Validate() error {
	if e.ID == "" {
		return errors.New("event id must not be empty")
	}
	if e.Timestamp == "" {
		return errors.New("event timestamp must not be empty")
	}
	if _, err := time.Parse(time.RFC3339, e.Timestamp); err != nil {
		return fmt.Errorf("event timestamp must be RFC3339: %w", err)
	}
	if e.Severity == "" {
		return errors.New("event severity must not be empty")
	}
	if e.Message == "" {
		return errors.New("event message must not be empty")
	}
	return nil
}

// SendEvent transmits a single event with retry handling.
func (c *Client) SendEvent(ctx context.Context, event Event) error {
	if err := event.Validate(); err != nil {
		return err
	}
	payload, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}
	policy := c.retryPolicy()
	var lastErr error
	for attempt := 1; attempt <= policy.MaxAttempts; attempt++ {
		req, reqErr := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint, bytes.NewReader(payload))
		if reqErr != nil {
			return fmt.Errorf("build request: %w", reqErr)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Idempotency-Key", event.ID)

		resp, doErr := c.httpClient().Do(req)
		if doErr == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				c.logger().Printf("sent security event %s", event.ID)
				return nil
			}
			if resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests {
				doErr = fmt.Errorf("retryable status %d", resp.StatusCode)
			} else {
				return fmt.Errorf("non-retryable status %d", resp.StatusCode)
			}
		}

		if errors.Is(doErr, context.Canceled) || errors.Is(doErr, context.DeadlineExceeded) {
			return doErr
		}

		lastErr = doErr
		if attempt == policy.MaxAttempts {
			break
		}
		backoff := policy.backoff(attempt)
		c.logger().Printf(
			"retrying event %s after error: %v (backoff=%s)",
			event.ID,
			doErr,
			backoff,
		)
		select {
		case <-time.After(backoff):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return fmt.Errorf("send event failed after %d attempts: %w", policy.MaxAttempts, lastErr)
}

// Stream consumes events from the channel until it is closed or the context is cancelled.
func (c *Client) Stream(ctx context.Context, events <-chan Event) error {
	var errs []error
	for {
		select {
		case <-ctx.Done():
			if len(errs) > 0 {
				return errors.Join(append(errs, ctx.Err())...)
			}
			return ctx.Err()
		case event, ok := <-events:
			if !ok {
				if len(errs) > 0 {
					return errors.Join(errs...)
				}
				return nil
			}
			if err := c.SendEvent(ctx, event); err != nil {
				wrapped := fmt.Errorf("event %s: %w", event.ID, err)
				errs = append(errs, wrapped)
				c.logger().Printf("failed to send event %s: %v", event.ID, err)
			}
		}
	}
}

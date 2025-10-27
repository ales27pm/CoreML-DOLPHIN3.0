package securitylog

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

type capturedEvent struct {
	Event
	Attempts int32
}

func newTestClient(endpoint string) *Client {
	return &Client{
		Endpoint:   endpoint,
		HTTPClient: &http.Client{Timeout: 2 * time.Second},
		Logger:     log.New(io.Discard, "", log.LstdFlags),
		RetryPolicy: RetryPolicy{
			MaxAttempts:    4,
			InitialBackoff: 10 * time.Millisecond,
			MaxBackoff:     40 * time.Millisecond,
		},
	}
}

func TestSendEventSuccess(t *testing.T) {
	var received capturedEvent
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&received.Attempts, 1)
		defer r.Body.Close()
		if r.Header.Get("Content-Type") != "application/json" {
			t.Fatalf("expected JSON content-type, got %s", r.Header.Get("Content-Type"))
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode event: %v", err)
		}
		w.WriteHeader(http.StatusAccepted)
	}))
	defer server.Close()

	client := newTestClient(server.URL)
	event := Event{ID: "evt-1", Timestamp: time.Now().UTC().Format(time.RFC3339), Severity: "HIGH", Message: "intrusion detected"}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if err := client.SendEvent(ctx, event); err != nil {
		t.Fatalf("SendEvent returned error: %v", err)
	}
	if received.ID != event.ID {
		t.Fatalf("expected id %s, got %s", event.ID, received.ID)
	}
	if received.Attempts != 1 {
		t.Fatalf("expected one attempt, got %d", received.Attempts)
	}
}

func TestSendEventRetriesOnServerError(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		current := attempts.Add(1)
		defer r.Body.Close()
		if current < 3 {
			http.Error(w, "try again", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := newTestClient(server.URL)
	event := Event{ID: "evt-2", Timestamp: time.Now().UTC().Format(time.RFC3339), Severity: "MEDIUM", Message: "suspicious login"}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := client.SendEvent(ctx, event); err != nil {
		t.Fatalf("SendEvent should retry until success, got error: %v", err)
	}
	if attempts.Load() < 3 {
		t.Fatalf("expected at least three attempts, got %d", attempts.Load())
	}
}

func TestStreamStopsOnContextCancel(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer server.Close()

	client := newTestClient(server.URL)
	events := make(chan Event, 1)
	events <- Event{ID: "evt-3", Timestamp: time.Now().UTC().Format(time.RFC3339), Severity: "LOW", Message: "scan completed"}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	err := client.Stream(ctx, events)
	if err == nil || err != context.Canceled {
		t.Fatalf("expected context cancellation error, got %v", err)
	}
	if attempts.Load() == 0 {
		t.Fatalf("expected at least one event to be sent")
	}
}

func TestSendEventValidatesInput(t *testing.T) {
	client := newTestClient("http://example.com")
	ctx := context.Background()
	err := client.SendEvent(ctx, Event{})
	if err == nil {
		t.Fatalf("expected validation error for empty event")
	}
}

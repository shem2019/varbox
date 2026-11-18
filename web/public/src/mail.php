<?php
declare(strict_types=1);

// Outbound email for the site. The only message it sends today tells Shem a recording is ready to
// process. Settings live under `mail` in storage/config.json; without them the message is logged.

/** Append a line to storage/mail.log, so an unsent message leaves a trace. */
function mail_log(string $message): void
{
    $line = '[' . now() . '] ' . $message . "\n";
    @file_put_contents(storage_path('mail.log'), $line, FILE_APPEND);
}

/**
 * Send one plain-text email through the SMTP server in config.
 *
 * Returns true when the server accepted the message. With no `mail.host` configured it logs the
 * message and returns false, so callers can carry on regardless.
 */
function send_mail(string $subject, string $body, ?string $to = null): bool
{
    $cfg = (array) (config()['mail'] ?? []);
    $host = (string) ($cfg['host'] ?? '');
    $to = $to ?? (string) ($cfg['to'] ?? '');
    $from = (string) ($cfg['from'] ?? ($cfg['user'] ?? ''));
    if ($host === '' || $to === '' || $from === '') {
        mail_log("not configured; would send to $to: $subject");
        return false;
    }
    try {
        smtp_send($cfg, $from, $to, $subject, $body);
        mail_log("sent to $to: $subject");
        return true;
    } catch (Throwable $e) {
        mail_log("send failed ($to: $subject): " . $e->getMessage());
        return false;
    }
}

/** Talk SMTP to the configured server. Throws on any protocol or connection error. */
function smtp_send(array $cfg, string $from, string $to, string $subject, string $body): void
{
    $host = (string) $cfg['host'];
    $port = (int) ($cfg['port'] ?? 587);
    $secure = (string) ($cfg['secure'] ?? ($port === 465 ? 'ssl' : 'tls'));
    $timeout = 20;

    $transport = $secure === 'ssl' ? "ssl://$host:$port" : "$host:$port";
    $fp = @stream_socket_client($transport, $errno, $errstr, $timeout);
    if (!$fp) {
        throw new RuntimeException("connect $host:$port failed: $errstr");
    }
    stream_set_timeout($fp, $timeout);

    $read = function () use ($fp): string {
        $data = '';
        while (($line = fgets($fp, 1024)) !== false) {
            $data .= $line;
            // The last line of a reply has a space (not a dash) after the three-digit code.
            if (strlen($line) < 4 || $line[3] === ' ') {
                break;
            }
        }
        return $data;
    };
    $expect = function (string $reply, string $ok) {
        if (strncmp($reply, $ok, strlen($ok)) !== 0) {
            throw new RuntimeException('SMTP said: ' . trim($reply));
        }
    };
    $cmd = function (string $line, string $ok) use ($fp, $read, $expect): void {
        fwrite($fp, $line . "\r\n");
        $expect($read(), $ok);
    };

    $expect($read(), '220');
    $ehlo = 'EHLO ' . (gethostname() ?: 'varbox');
    $cmd($ehlo, '250');
    if ($secure === 'tls') {
        $cmd('STARTTLS', '220');
        if (!stream_socket_enable_crypto($fp, true, STREAM_CRYPTO_METHOD_TLS_CLIENT)) {
            throw new RuntimeException('STARTTLS failed');
        }
        $cmd($ehlo, '250');
    }
    if (!empty($cfg['user'])) {
        $cmd('AUTH LOGIN', '334');
        $cmd(base64_encode((string) $cfg['user']), '334');
        $cmd(base64_encode((string) ($cfg['pass'] ?? '')), '235');
    }
    $cmd('MAIL FROM:<' . $from . '>', '250');
    $cmd('RCPT TO:<' . $to . '>', '250');
    $cmd('DATA', '354');
    $headers = [
        'From: ' . $from,
        'To: ' . $to,
        'Subject: ' . $subject,
        'Date: ' . gmdate('D, d M Y H:i:s') . ' +0000',
        'MIME-Version: 1.0',
        'Content-Type: text/plain; charset=utf-8',
    ];
    // Dot-stuff lines that begin with a dot, then end the message with a lone dot.
    $data = preg_replace('/^\./m', '..', str_replace("\r\n", "\n", $body));
    $data = str_replace("\n", "\r\n", (string) $data);
    fwrite($fp, implode("\r\n", $headers) . "\r\n\r\n" . $data . "\r\n.\r\n");
    $expect($read(), '250');
    $cmd('QUIT', '221');
    fclose($fp);
}

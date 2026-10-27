// Sign-in for the RetiVASC server, and the account state the app shows.
// "offline" means no server answered: the canvas then runs as the browser-only demo.

import { useEffect, useState, type FormEvent } from 'react';
import { api, ServerError, type Account } from '../server/api';

export type AccountState = { mode: 'checking' } | { mode: 'offline' } | { mode: 'signin' } | { mode: 'ready'; account: Account };

export function useAccount(): [AccountState, (s: AccountState) => void] {
  const [state, setState] = useState<AccountState>({ mode: 'checking' });
  useEffect(() => {
    api.me().then(
      (account) => setState({ mode: 'ready', account }),
      (e: ServerError) => setState({ mode: e.status === 401 ? 'signin' : 'offline' }),
    );
  }, []);
  return [state, setState];
}

export function Login({ onDone }: { onDone: (a: Account) => void }) {
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      onDone(await api.login(name, password));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="login-wrap">
      <form className="login" onSubmit={submit}>
        <h2>RetiVASC</h2>
        <p className="faint">Sign in to analyse photos on this server.</p>
        <label>User name<input autoFocus value={name} onChange={(e) => setName(e.target.value)} autoComplete="username" /></label>
        <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" /></label>
        {error && <div className="error">{error}</div>}
        <button className="btn primary" disabled={busy || !name || !password}>{busy ? 'Signing in…' : 'Sign in'}</button>
        <p className="hint">Photos and results stay on this server and are deleted after the retention period.</p>
      </form>
    </div>
  );
}

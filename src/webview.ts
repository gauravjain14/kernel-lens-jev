import * as vscode from 'vscode';
import { readFileSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import type { LiveState } from './live-types';

export class LensView implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;
  constructor(private context: vscode.ExtensionContext, private getState: () => LiveState, private onMessage: (message: unknown) => Promise<void>) {}
  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    const root = vscode.Uri.joinPath(this.context.extensionUri, 'media');
    view.webview.options = { enableScripts: true, localResourceRoots: [root] };
    const asset = (name: string) => view.webview.asWebviewUri(vscode.Uri.joinPath(root, name)).toString();
    const nonce = randomBytes(18).toString('base64');
    view.webview.html = readFileSync(vscode.Uri.joinPath(root, 'panel.html').fsPath, 'utf8')
      .replaceAll('__CSP__', view.webview.cspSource).replaceAll('__NONCE__', nonce)
      .replaceAll('__STYLE__', asset('panel.css')).replaceAll('__SCRIPT__', asset('panel.js'));
    const listener = view.webview.onDidReceiveMessage(message => {
      if (message?.type === 'ready') this.update();
      else void this.onMessage(message).catch(() => vscode.window.showErrorMessage('Kernel Lens could not complete that action.'));
    });
    view.onDidDispose(() => { listener.dispose(); this.view = undefined; });
    this.update();
  }
  update(): void { void this.view?.webview.postMessage({ type: 'state', state: this.getState() }); }
}

# Installing, distributing, and publishing Kernel Lens

## Fastest distribution: a VSIX

Build the extension with `npm ci`, `npm run check`, and `npm run package`. Share `artifacts/kernel-lens-0.6.0.vsix` through a repository release or your team's normal file-sharing channel.

Each recipient uses Extensions → `…` → **Install from VSIX…**, then opens a trusted workspace, connects their own Vercel AI Gateway key once, and starts typing. Jev assessments run automatically after setup; Astra reviews are on demand by default. This requires no backend, hosting account for your application, or deployed Next.js project. The external services are Vercel AI Gateway, Jev, and the selected model when a generated review is requested.

The CLI equivalent is:

```bash
code --install-extension kernel-lens-0.6.0.vsix
```

VSIX compatibility with other VS Code-based editors depends on that editor's extension API version. The current minimum is VS Code 1.98. Test those editors separately before advertising compatibility.

## Visual Studio Marketplace

These are publishing instructions; no publisher account or listing has been created by this project.

1. Create a publisher at https://marketplace.visualstudio.com/manage and keep its publisher ID.
2. Configure your real publisher and public repository in `package.json`:

   ```bash
   npm pkg set publisher=YOUR_PUBLISHER_ID
   npm pkg set repository.type=git repository.url=https://github.com/YOUR_ACCOUNT/kernel-lens.git
   ```

3. Review the public name, description, README, license, and current version. The source currently uses `kernel-lens-local` solely for local VSIX packaging. Change it before publishing. Changing publisher later changes the extension identity.
4. Create the publishing credential described in Microsoft's current guide. For a personal access token, use the required Marketplace Manage permission. Store the token outside the repository.
5. Authenticate and publish:

   ```bash
   npx @vscode/vsce login YOUR_PUBLISHER_ID
   npm run check
   npx @vscode/vsce publish --no-dependencies
   ```

The resulting public listing can be shared as `https://marketplace.visualstudio.com/items?itemName=YOUR_PUBLISHER_ID.kernel-lens`. Public availability depends on Marketplace validation. Recipients then install from the Extensions search and receive Marketplace updates.

Official publishing guide: https://code.visualstudio.com/api/working-with-extensions/publishing-extension

## Updates

Update the version in `package.json`, adjust the artifact filename in the package script, run checks, and package/publish the new version. For private VSIX distribution, recipients install the new file manually. Keep the same publisher and extension name to replace the existing installation.

The included GitHub Actions workflow tests, builds, and produces a downloadable VSIX for pushes and manual runs. It intentionally has no publishing credential or automatic public release step. The default local publisher is suitable for internal trials; settle on the final identity before broad distribution.

## Before a wider release

Run `npm run eval:live` with your own key and add labeled examples from actual CUDA, inference, and post-training work. Review both missed issues and false alarms. Test an editor with Remote SSH if that is how your users access GPU machines. Feedback distinguishes Jev predictions from generated reviews; compiler checks, unit tests, compute-sanitizer, and profiler measurements remain the evidence for program behavior and performance.

The beta includes no telemetry or central subscription system. If a future hosted service provides shared credentials, implement server-side authentication and usage accounting; do not place a shared provider key inside a public VSIX.

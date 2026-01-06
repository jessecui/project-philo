export default function HomePage() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="max-w-4xl mx-auto p-8 space-y-4">
        <h1 className="text-4xl font-bold text-foreground">
          Project Philo
        </h1>
        <p className="text-muted-foreground">
          Upgraded to Next.js 15, React 19, and Tailwind CSS v4
        </p>
      </div>
    </div>
  );
}

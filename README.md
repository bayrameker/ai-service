# AI Service - Bağımsız AI ve Agentic AI Hizmet Platformu

## 🚀 Genel Bakış

Bu proje, sistemlere ve yapılara kapsamlı AI ve Agentic AI hizmeti veren bağımsız bir platform geliştirmek için tasarlanmıştır. Platform, modern LLM sağlayıcıları, agentic mimariler, self-learning sistemler ve multi-agent collaboration özelliklerini içerir.

## 🏗️ Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Service Platform                      │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer (React/Next.js)                            │
├─────────────────────────────────────────────────────────────┤
│  API Gateway (FastAPI/Express)                             │
├─────────────────────────────────────────────────────────────┤
│  Agent Orchestration Layer                                 │
│  ├── Agent Manager                                         │
│  ├── Task Queue (Redis/RabbitMQ)                          │
│  └── Workflow Engine                                       │
├─────────────────────────────────────────────────────────────┤
│  Core Services                                             │
│  ├── LLM Provider Manager                                  │
│  ├── Memory System (Vector DB + PostgreSQL)               │
│  ├── Agent Collaboration (A2A Protocol)                   │
│  └── Security & Auth                                       │
├─────────────────────────────────────────────────────────────┤
│  Integration Layer                                         │
│  ├── 3rd Party APIs                                        │
│  ├── Database Connectors                                   │
│  └── Cloud Services                                        │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Teknoloji Stack

### Backend
- **Framework**: FastAPI (Python) / Express.js (Node.js)
- **Agent Framework**: LangGraph + AutoGen
- **Database**: PostgreSQL + Redis
- **Vector DB**: FAISS / Chroma
- **Message Queue**: Redis / RabbitMQ

### LLM Providers
- **OpenAI**: GPT-4.1, GPT-4o, GPT-4.5 Preview
- **Anthropic**: Claude 4, Claude 3.5 Sonnet, Haiku
- **Local**: Ollama (Llama 3.3, DeepSeek R1)

### Frontend
- **Framework**: React with Next.js
- **UI Library**: Tailwind CSS + Shadcn/ui
- **State Management**: Zustand / Redux Toolkit

### DevOps
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (optional)
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions

## 📁 Proje Yapısı

```
ai-service/
├── backend/
│   ├── app/
│   │   ├── agents/          # Agent yönetimi
│   │   ├── llm_providers/   # LLM entegrasyonları
│   │   ├── memory/          # Hafıza sistemleri
│   │   ├── collaboration/   # Agent işbirliği
│   │   ├── integrations/    # 3. parti entegrasyonlar
│   │   ├── security/        # Güvenlik ve auth
│   │   └── tests/           # Test dosyaları
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/      # React bileşenleri
│   │   ├── pages/           # Sayfa bileşenleri
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # API servisleri
│   │   └── utils/           # Yardımcı fonksiyonlar
│   ├── package.json
│   └── Dockerfile
├── docs/                    # Dokümantasyon
├── scripts/                 # Deployment scriptleri
├── docker-compose.yml
└── README.md
```

## 🎯 Ana Özellikler

### 1. LLM Sağlayıcı Entegrasyonları
- OpenAI API entegrasyonu (GPT-4.1, GPT-4o)
- Anthropic Claude entegrasyonu (Claude 4, 3.5 Sonnet)
- Ollama lokal LLM desteği
- Birleşik LLM provider abstraction layer
- Dinamik model switching ve load balancing

### 2. Agentic Mimari
- LangGraph tabanlı agent framework
- Agent lifecycle management
- Role-based agent capabilities
- Task queue ve workflow engine
- Agent state management

### 3. AI Learning Memory Sistemi
- Vector database entegrasyonu (FAISS/Chroma)
- Episodic memory (geçmiş deneyimler)
- Semantic memory (bilgi ve kavramlar)
- Working memory management
- RAG (Retrieval Augmented Generation)
- Self-learning algorithms

### 4. Multi-Agent Collaboration
- A2A (Agent2Agent) protokol implementasyonu
- Agent discovery ve registry
- Message passing sistemi
- Collaboration patterns
- Conflict resolution
- Multi-agent orchestration

### 5. 3. Parti Entegrasyonlar
- RESTful API gateway
- Webhook sistemi
- Database connectors (SQL/NoSQL)
- Cloud services integration
- Third-party API adapters
- Plugin architecture

### 6. Güvenlik ve Kimlik Doğrulama
- Agent authentication
- Role-based access control (RBAC)
- API key management
- Encryption ve security protocols
- Audit logging
- Rate limiting ve throttling

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL
- Redis

### Kurulum

1. **Repository'yi klonlayın**
```bash
git clone <repository-url>
cd ai-service
```

2. **Backend kurulumu**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend kurulumu**
```bash
cd frontend
npm install
```

4. **Docker ile çalıştırma**
```bash
docker-compose up -d
```

## 📚 Dokümantasyon

Detaylı dokümantasyon için `docs/` klasörüne bakınız:
- [API Dokümantasyonu](docs/api.md)
- [Agent Geliştirme Kılavuzu](docs/agent-development.md)
- [Deployment Kılavuzu](docs/deployment.md)
- [Güvenlik Kılavuzu](docs/security.md)

## 🧪 Test

```bash
# Backend testleri
cd backend
pytest

# Frontend testleri
cd frontend
npm test

# Integration testleri
docker-compose -f docker-compose.test.yml up
```

## 📈 Roadmap

- [x] Temel mimari tasarımı
- [ ] LLM provider entegrasyonları
- [ ] Agent framework implementasyonu
- [ ] Memory sistemi geliştirme
- [ ] A2A protokol implementasyonu
- [ ] 3. parti entegrasyonlar
- [ ] Güvenlik sistemleri
- [ ] Kapsamlı test suite

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 📞 İletişim

Proje hakkında sorularınız için issue açabilir veya iletişime geçebilirsiniz.

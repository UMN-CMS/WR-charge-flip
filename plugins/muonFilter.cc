// -*- C++ -*-
 //
 // Package:    muonFilter
 // Class:      muonFilter
 //
 /* 
 
  Description: Filter to select events with gen muons with pT above a minimum threshhold, also requires an electron in the event.
 
      
 */
 //
 
 // system include files
 #include <memory>
 #include <iostream>
 #include <set>
 
 // user include files
 
 #include "FWCore/Framework/interface/Frameworkfwd.h"
 #include "FWCore/Framework/interface/global/EDFilter.h"
 
 #include "FWCore/Framework/interface/Event.h"
 #include "FWCore/Framework/interface/MakerMacros.h"
 
 #include "FWCore/ParameterSet/interface/ParameterSet.h"
 #include "DataFormats/HepMCCandidate/interface/GenParticle.h"
 
 //
 // class declaration
 //
 
 class muonFilter : public edm::global::EDFilter<> {
 public:
   explicit muonFilter(const edm::ParameterSet&);
   ~muonFilter() override;
 
   bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
 
 private:
 	bool tWfinder(const edm::Event&, const reco::GenParticle* ) const;
   // ----------member data ---------------------------
 
   double ptMin_;          // number of particles required to pass filter
   edm::EDGetToken m_genParticleToken;
 };
 
 // using namespace edm;
 // using namespace std;
 
 muonFilter::muonFilter(const edm::ParameterSet& iConfig)
     : 
     ptMin_(iConfig.getParameter<double>("ptMin")),
	 m_genParticleToken(consumes<std::vector<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genParticles")))
 {
   //here do whatever other initialization is needed
 }
 
 muonFilter::~muonFilter() {
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
 }
 
 // ------------ method called to skim the data  ------------
 bool muonFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

    bool highPT = false;
    bool electron = false;
    bool muon = false;
    int pdgId = 0;

   edm::Handle<std::vector<reco::GenParticle>> genParticles;
   iEvent.getByToken(m_genParticleToken, genParticles);
 
   for (std::vector<reco::GenParticle>::const_iterator iParticle = genParticles->begin(); iParticle != genParticles->end(); iParticle++) {
     if( ! iParticle->isHardProcess() ){ continue; }
     if( ! tWfinder(iEvent, &(*iParticle))){ continue;}
     pdgId = iParticle->pdgId();
     if (abs(pdgId) == 13) {
     	muon = true;
     	if (iParticle->pt() > ptMin_){
     		highPT = true;
     	}
     }
     else if(abs(pdgId) == 11){
     	electron = true;
     }
   }

return highPT & muon & electron;
}

bool muonFilter::tWfinder(const edm::Event& iEvent, const reco::GenParticle* lepton) const {

    		bool ttbar=false;
    		int iStatus;

    		const reco::Candidate* iParticle = lepton->mother();

    		while(iStatus!=4){  //status=4 is the initial proton
    			iStatus = iParticle->status();
    			
    			if(abs(iParticle->pdgId())==24){ //found W
    				while(iStatus!=4){
    				iParticle = iParticle->mother();
    				iStatus = iParticle->status();

    			   	if(abs(iParticle->pdgId())==6){ ttbar=true; 
    			   		break;
    			   	}
					}

    			}

    			iParticle = iParticle->mother();
    		}

		if(ttbar==true){return true;}
		else{return false;}
}

//define this as a plug-in
DEFINE_FWK_MODULE(muonFilter);
